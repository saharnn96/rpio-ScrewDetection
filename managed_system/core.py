"""Layer 2 (Application layer) - hardware-agnostic screw-detection primitives.

(Formerly named `screwSegmentation.py`; renamed to `detection_core.py` to
avoid clashing with the original `screw_detection/python/screwSegmentation.py`.)

This is the application layer of the three-layer architecture:

    Layer 3 (MAPLE-K nodes)   -->  maple_k.py
    Layer 2 (this file)       -->  ScrewDetectionCore + adapter INTERFACES
    Layer 1 (adapters)        -->  sim_adapters.py  |  real_adapters.py

Nothing here talks to hardware directly. Every camera/robot/model call is
funnelled through an injected adapter that satisfies the interfaces documented
below. Layer 3 talks to this file only. Adapters are chosen by whichever entry
point wires the core (`simulation.py` or `real_main.py`).
"""

import enum
import os
import shutil
import time
from collections import namedtuple


# ---------------------------------------------------------------------------
# Domain types used by every layer.
# ---------------------------------------------------------------------------
# Values are the trained YOLO class ids (same order as the original
# python/screwSegmentation.py): 0=holder, 1=noscrew, 2=screw. RealDetector
# passes model labels through unchanged, so these MUST stay in model order.
class DetectionClasses(enum.Enum):
    HOLDER = 0
    NOSCREW = 1
    SCREW = 2


class DetectionModels(enum.Enum):
    PC_CLOSEUP = 1
    SCREW_FIXTURE = 2
    SCREEN_AND_SCREEN_FRAME = 3
    PC_TOPVIEW = 4


DetectionResult = namedtuple(
    "DetectionResult", ["label", "box", "score", "mask", "entropy"]
)


def filter_detections_within_holders(detection_results):
    """Keep holders; keep screw/noscrew detections only when their bbox center
    lies inside a detected holder. Any other label is dropped.

    Port of the original filter_out_detections_not_within_holders() from
    python/screwSegmentation.py.
    """
    holders = [
        r for r in detection_results
        if r.label == DetectionClasses.HOLDER.value and r.box is not None
    ]
    filtered = []
    for result in detection_results:
        if result.label == DetectionClasses.HOLDER.value:
            filtered.append(result)
        elif result.label in (DetectionClasses.SCREW.value,
                              DetectionClasses.NOSCREW.value):
            center_x = (result.box[0] + result.box[2]) / 2
            center_y = (result.box[1] + result.box[3]) / 2
            for holder in holders:
                hx1, hy1, hx2, hy2 = holder.box
                if hx1 <= center_x <= hx2 and hy1 <= center_y <= hy2:
                    filtered.append(result)
                    break
    return filtered


# ---------------------------------------------------------------------------
# Adapter interfaces - the contract Layer 1 must satisfy.
#
# There is no abstract base class on purpose (Python duck-typing is enough).
# Any object whose attributes match these signatures can be injected.
# ---------------------------------------------------------------------------
#
#   class CameraAdapter:
#       def get_color_image(self) -> np.ndarray  # BGR uint8, HxWx3
#       def get_depth_snapshot(self) -> dict | None      # OPTIONAL; aligned to
#           # the last color frame: {"depth_image": np.ndarray uint16 HxW,
#           #  "fx","fy","ppx","ppy": float, "depth_units": float}
#           # Required only if the coord-transform helpers are used.
#
#   class RTDEAdapter:
#       def get_tcp_pose(self) -> dict           # {"x","y","z","rx","ry","rz"}
#
#   class DetectorAdapter:
#       model_id: int                            # currently deployed model
#       def detect(self, image, model_id) -> tuple[list[DetectionResult],
#                                                  float | None]
#           # returns (all detections, screw/noscrew entropy or None)
#
# ---------------------------------------------------------------------------


class ScrewDetectionCore:
    """Application-layer primitives that the MAPLE-K nodes call into.

    Hardware access is fully abstracted behind the three injected adapters.
    Swap sim adapters for real ones without touching this file or Layer 3.
    """

    def __init__(self, camera, rtde, detector, out_dir="./_sim_out",
                 entropy_window_size=10, entropy_threshold=0.5,
                 candidate_model_id=2, max_replans=3, calib_folder=None):
        self.camera = camera
        self.rtde = rtde
        self.detector = detector

        self.out_dir = out_dir
        self.rolling_dir = os.path.join(out_dir, "entropy_rolling_average_images")
        self.capture_dir = os.path.join(out_dir, "real_time_detection_images")
        os.makedirs(self.rolling_dir, exist_ok=True)
        os.makedirs(self.capture_dir, exist_ok=True)

        self.entropy_window_size = entropy_window_size
        self.entropy_threshold = entropy_threshold
        self.candidate_model_id = candidate_model_id
        self.max_replans = max_replans

        self.knowledge_log_path = os.path.join(out_dir, "knowledge_log.csv")

        # Hand-eye calibration (needed only for the coord-transform helpers).
        # `calib_folder` is the CalibData directory that holds
        # FinalTransforms/T_cam2gripper_Method_1.npz - same layout as the
        # original python/screwSegmentation.py expects.
        self.calib_folder = calib_folder
        self.T_cam2gripper = (
            self._load_T_cam2gripper(calib_folder) if calib_folder else None
        )

        # Depth snapshot taken alongside the last captured color frame
        # (populated by `capture()` when the camera adapter supports it).
        self.last_depth = None

    # --- cv2 lazy import so the module works without OpenCV ----------------
    @staticmethod
    def _cv2():
        import cv2
        return cv2

    # --- MONITOR primitive -------------------------------------------------
    def capture(self, detection_type):
        """Grab a frame from the injected camera and persist it to disk.

        Returns a JSON-friendly reference the Monitor node can drop into
        the FrameRef knowledge object.
        """
        cv2 = self._cv2()
        frame = self.camera.get_color_image()
        # Snapshot the aligned depth frame taken with this color frame so the
        # coord-transform helpers work on data from the same instant.
        get_depth = getattr(self.camera, "get_depth_snapshot", None)
        self.last_depth = get_depth() if get_depth is not None else None
        timestamp = time.strftime("%Y%m%d_%H%M%S") + f"_{int(time.time()*1000) % 1000:03d}"
        frame_path = os.path.join(
            self.capture_dir, f"new_image_{detection_type}_{timestamp}.jpg"
        )
        cv2.imwrite(frame_path, frame)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        brightness = float(gray.mean())
        return {
            "frame_path": frame_path,
            "timestamp": timestamp,
            "brightness": brightness,
        }

    # --- model routing policy ---------------------------------------------
    #
    # Two mechanisms decide which model runs:
    #   * The *task* (detection_type) picks a base model - topview / screen /
    #     fixture / closeup - exactly like the original change_active_model().
    #   * The MAPLE-K entropy loop only adapts the CLOSEUP task, swapping
    #     between its lighting-calibrated variants (ids 1 <-> 2).
    #
    # A detector MAY expose `ADAPTIVE_DETECTION_TYPES` (set[str]) and
    # `DETECTION_TYPE_TO_MODEL` (dict[str,int]). The SimulatedDetector exposes
    # neither, so the sim keeps its old behaviour: every frame is "adaptive"
    # and always uses the MAPLE-K-owned active model id.
    def is_adaptive(self, detection_type):
        """True if the MAPLE-K swap loop should run for this detection type."""
        adaptive = getattr(self.detector, "ADAPTIVE_DETECTION_TYPES", None)
        return adaptive is None or detection_type in adaptive

    def resolve_model_id(self, detection_type, active_model_id):
        """Which model id to detect with for this task.

        Closeup (adaptive) -> whatever MAPLE-K currently has active (1 or 2).
        Other tasks        -> the fixed model bound to that detection type.
        """
        if self.is_adaptive(detection_type):
            return active_model_id
        mapping = getattr(self.detector, "DETECTION_TYPE_TO_MODEL", {})
        return mapping.get(detection_type, active_model_id)

    # --- ANALYZE / LEGITIMATE primitive -----------------------------------
    def detect(self, frame_path, model_id):
        """Run detection + UQ through the injected detector.

        The image is read from disk (not passed through the knowledge store)
        so this same method serves both Analyze (live frame) and Legitimate
        (rolling frames re-scored with a candidate model).
        """
        cv2 = self._cv2()
        image = cv2.imread(frame_path)
        return self.detector.detect(image, model_id)

    def append_rolling_image(self, frame_path):
        """Keep only the last `entropy_window_size` frames in the rolling dir."""
        dst = os.path.join(self.rolling_dir, os.path.basename(frame_path))
        shutil.copyfile(frame_path, dst)
        images = sorted(os.listdir(self.rolling_dir))
        while len(images) > self.entropy_window_size:
            os.remove(os.path.join(self.rolling_dir, images.pop(0)))

    def list_rolling_images(self):
        return [
            os.path.join(self.rolling_dir, n)
            for n in sorted(os.listdir(self.rolling_dir))
        ]

    # --- coordinate transforms (ported from python/screwSegmentation.py) ---
    #
    # Pipeline: image point (px) -> camera frame (m, via aligned depth +
    # intrinsics) -> robot base frame (via TCP pose + hand-eye calibration).
    # These are what `get_detected_object_coords` on the XMLRPC bridge uses to
    # answer the UR pendant.

    @staticmethod
    def _load_T_cam2gripper(calib_folder):
        import numpy as np
        path = os.path.join(calib_folder, "FinalTransforms",
                            "T_cam2gripper_Method_1.npz")
        return np.load(path)["arr_0"]

    def get_translation_T_cam2gripper(self):
        """xyz translation of the hand-eye calibration (pendant RPC helper)."""
        if self.T_cam2gripper is None:
            raise RuntimeError("No calibration loaded (calib_folder not set).")
        return [float(x) for x in self.T_cam2gripper[:3, 3]]

    def pose_to_transformation_matrix(self, pose):
        """[x,y,z,rx,ry,rz] (axis-angle) -> 4x4 homogeneous transform."""
        import numpy as np
        cv2 = self._cv2()
        pose = np.asarray(pose, dtype=float)
        tvec = pose[:3]
        rvec = pose[3:]
        R = cv2.Rodrigues(rvec)[0]
        T = np.concatenate((R, tvec[:, None]), axis=1)
        T = np.concatenate((T, np.array([[0, 0, 0, 1]])), axis=0)
        return T

    def img_point_to_cam_point(self, image_point, view="close"):
        """2D image point -> 3D point (m) in the camera frame.

        Uses the aligned depth image snapshotted at capture() time. Zero-depth
        pixels fall back to per-view defaults; 'close' always uses the fixed
        measured height - identical to the original implementation.
        """
        import numpy as np
        if self.last_depth is None:
            raise RuntimeError(
                "No depth snapshot available - the camera adapter must provide "
                "get_depth_snapshot() for coordinate transforms."
            )
        depth = self.last_depth
        depth_value = depth["depth_image"][int(image_point[1]), int(image_point[0])]
        if view == "highest":
            if depth_value == 0:
                depth_value = 550
        elif view == "top":
            if depth_value == 0:
                depth_value = 350
        elif view == "close":
            depth_value = 165
            # Measured height from cam bottom to hole at each corner: 152-153

        # Convert the 2D image point to 3D camera coordinates
        z = depth_value * depth["depth_units"]
        x = (image_point[0] - depth["ppx"]) * z / depth["fx"]
        y = (image_point[1] - depth["ppy"]) * z / depth["fy"]
        return np.array([x, y, z])

    def calc_img_point_to_base_frame(self, image_point, tcp_pose, view="close"):
        """Image point (px) -> 3D point (m) in the robot base frame.

        `tcp_pose` is the robot pose [x,y,z,rx,ry,rz] at the moment the image
        was taken (the pendant passes it into take_new_image_and_detect).
        """
        import numpy as np
        if self.T_cam2gripper is None:
            raise RuntimeError("No calibration loaded (calib_folder not set).")

        # Check the image point is within the frame
        img_h, img_w = self.last_depth["depth_image"].shape[:2]
        if (image_point[0] < 0 or image_point[0] >= img_w
                or image_point[1] < 0 or image_point[1] >= img_h):
            print("Image point is outside the image frame")
            return None

        T_base2gripper = self.pose_to_transformation_matrix(tcp_pose)
        cam_point = self.img_point_to_cam_point(image_point, view)
        P_cam = np.append(cam_point, [1])

        # Camera frame expressed in the base frame, then the point itself.
        T_base2cam = T_base2gripper @ np.linalg.inv(self.T_cam2gripper)
        cam_coord_in_base_frame = T_base2cam @ P_cam
        return cam_coord_in_base_frame[:3]

    def get_coordinates_list(self, bbox_centers, tcp_pose, camera_view="close"):
        """Transform bbox centers to base-frame coords, sorted and flattened
        exactly like the original (sort by y with tolerance, then x)."""
        coordinates_list = []
        for center in bbox_centers:
            coord = self.calc_img_point_to_base_frame(center, tcp_pose, view=camera_view)
            if coord is not None:
                coordinates_list.append([float(x) for x in coord])

        tolerance = 0.1
        coordinates_list = sorted(
            coordinates_list,
            key=lambda c: (round(c[1] / tolerance) * tolerance, c[0]),
        )
        # flatten [[x,y,z], ...] -> [x,y,z,x,y,z,...] (URScript-friendly)
        return [item for sublist in coordinates_list for item in sublist]

    # --- KNOWLEDGE primitive ----------------------------------------------
    def log_knowledge(self, timestamp, frame_ref, detections):
        import csv
        new_file = not os.path.exists(self.knowledge_log_path)
        with open(self.knowledge_log_path, "a", newline="") as f:
            writer = csv.writer(f)
            if new_file:
                writer.writerow(["timestamp", "brightness", "active_model",
                                 "num_detections", "screw_entropy"])
            screw_entropy = next(
                (d["entropy"] for d in detections
                 if d["label"] in (DetectionClasses.SCREW.value,
                                   DetectionClasses.NOSCREW.value)
                 and d["entropy"] is not None),
                None,
            )
            writer.writerow([
                timestamp,
                f"{frame_ref.get('brightness', 0):.2f}",
                self.detector.model_id,
                len(detections),
                f"{screw_entropy:.4f}" if screw_entropy is not None else "",
            ])


# ---------------------------------------------------------------------------
# Shared-instance singleton.
#
# The entry point (simulation.py or real_main.py) builds the core with the
# adapters it wants and calls set_core(core) once. Every MAPLE-K node then
# reads it via `detection_core.CORE`. Nodes must NOT `from detection_core
# import CORE` (that binds the name at import time when CORE is still None) -
# they must reference the module attribute so it sees the value set later.
# ---------------------------------------------------------------------------
CORE: "ScrewDetectionCore | None" = None


def set_core(core):
    global CORE
    CORE = core
