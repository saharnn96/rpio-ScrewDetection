"""Layer 2 (Application layer) - hardware-agnostic screw-detection primitives.

This is the "screwSegmentation layer" of the three-layer architecture:

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
class DetectionClasses(enum.Enum):
    HOLDER = 0
    SCREW = 1
    NOSCREW = 2


class DetectionModels(enum.Enum):
    PC_CLOSEUP = 1
    SCREW_FIXTURE = 2
    SCREEN_AND_SCREEN_FRAME = 3
    PC_TOPVIEW = 4


DetectionResult = namedtuple(
    "DetectionResult", ["label", "box", "score", "mask", "entropy"]
)


# ---------------------------------------------------------------------------
# Adapter interfaces - the contract Layer 1 must satisfy.
#
# There is no abstract base class on purpose (Python duck-typing is enough).
# Any object whose attributes match these signatures can be injected.
# ---------------------------------------------------------------------------
#
#   class CameraAdapter:
#       def get_color_image(self) -> np.ndarray  # BGR uint8, HxWx3
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
                 candidate_model_id=2, max_replans=3):
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
# reads it via `screwSegmentation.CORE`. Nodes must NOT `from screwSegmentation
# import CORE` (that binds the name at import time when CORE is still None) -
# they must reference the module attribute so it sees the value set later.
# ---------------------------------------------------------------------------
CORE: "ScrewDetectionCore | None" = None


def set_core(core):
    global CORE
    CORE = core
