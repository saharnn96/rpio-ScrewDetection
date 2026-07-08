"""Layer 1 (Hardware layer) - REAL adapters for field deployment.

These wrap the same libraries the original `python/screwSegmentation.py` uses:
  * pyrealsense2       (color+depth camera stream)
  * RDTEReceive.RTDEReceive  (UR robot telemetry)
  * ultralytics.YOLO + inference_uq_single_detection.run_uq  (detector + UQ)

All heavy imports are done lazily inside `__init__` / `detect`, so this module
imports cleanly on a machine that only has the sim libraries. Real deployment
needs the extras listed at the bottom of `requirements.txt`.

TODO for the field:
  * Fill in the hardcoded model paths in `RealDetector.MODEL_PATHS`.
  * The interfaces here must match those documented in `detection_core.py`.
"""

import os
import sys

import numpy as np

from detection_core import (
    DetectionResult, DetectionClasses, filter_detections_within_holders,
)

# Absolute path to this file's directory (repo root). Everything the real
# adapters load - vendored RDTEReceive config, model weights - is resolved
# against this so `real_main.py` works no matter what CWD the UR controller
# launches it from.
_HERE = os.path.dirname(os.path.abspath(__file__))

# The original code these adapters wrap (the RDTEReceive package and
# inference_uq_single_detection) lives under screw_detection/python and is
# written to be imported from that directory - put it on sys.path so the
# lazy imports below resolve from any CWD.
_ORIG_PYTHON_DIR = os.path.join(_HERE, "screw_detection", "python")
if _ORIG_PYTHON_DIR not in sys.path:
    sys.path.append(_ORIG_PYTHON_DIR)
_DETECTION_MODEL_DIR = os.path.join(
    _HERE, "screw_detection", "detection_model"
)


def _detect_weights(*parts):
    """Absolute path to a detect-run best.pt under the repo's detection_model."""
    return os.path.join(_DETECTION_MODEL_DIR, "scripts", "runs", "detect",
                        *parts, "weights", "best.pt")


# ---------------------------------------------------------------------------
# CameraAdapter - RealSense pipeline (aligned color+depth).
# ---------------------------------------------------------------------------
class RealCamera:
    """Wraps the RealSense init/frame-grab logic from the original code."""

    def __init__(self, width=1280, height=720, exposure_us=2500):
        import pyrealsense2 as rs

        self._rs = rs
        self.pipeline = rs.pipeline()
        cfg = rs.config()
        cfg.enable_stream(rs.stream.color, width, height, rs.format.bgr8, 30)
        cfg.enable_stream(rs.stream.depth, width, height, rs.format.z16, 30)
        self.align = rs.align(rs.stream.color)
        self.pipeline.start(cfg)

        # Warm-up and exposure lock.
        self.pipeline.wait_for_frames(5000)
        self._sensor = self.pipeline.get_active_profile().get_device().query_sensors()[1]
        self._sensor.set_option(rs.option.exposure, exposure_us)
        self.exposure_us = exposure_us

        self.last_depth_frame = None  # aligned depth of the last color frame

    def get_color_image(self):
        frames = self.align.process(self.pipeline.wait_for_frames(5000))
        color = frames.get_color_frame()
        self.last_depth_frame = frames.get_depth_frame()
        return np.asanyarray(color.get_data())

    def get_depth_snapshot(self):
        """Depth data aligned to the last color frame, in the plain-dict form
        `detection_core`'s coord-transform helpers consume."""
        df = self.last_depth_frame
        if df is None:
            return None
        intr = df.profile.as_video_stream_profile().intrinsics
        return {
            "depth_image": np.asanyarray(df.get_data()),
            "fx": intr.fx, "fy": intr.fy,
            "ppx": intr.ppx, "ppy": intr.ppy,
            "depth_units": df.get_units(),
        }

    # --- exposure control (pendant RPC surface) ----------------------------
    def get_exposure(self):
        return self.exposure_us

    def set_exposure(self, exposure_us):
        import time
        self.exposure_us = exposure_us
        self._sensor.set_option(self._rs.option.exposure, exposure_us)
        print(f"Exposure time set to: {exposure_us}")
        time.sleep(0.2)  # let the camera adjust


# ---------------------------------------------------------------------------
# RTDEAdapter - UR robot telemetry.
# ---------------------------------------------------------------------------
class RealRTDE:
    """Wraps RTDEReceive. The original code streams tool_current and
    runtime_state too; only the TCP pose is required by the loop."""

    def __init__(self, robot_ip, config_file=None):
        # Import path mirrors what the original screwSegmentation.py uses.
        from RDTEReceive.RTDEReceive import RTDEReceive

        # RTDEReceive loads its recipe file relative to CWD by default; anchor
        # it to the vendored copy next to this module so it resolves regardless
        # of where the process was launched from.
        if config_file is None:
            config_file = os.path.join(_HERE, "record_configuration.xml")

        self._rtde = RTDEReceive(robot_ip, config_file=config_file)
        self._rtde.connect_to_robot()
        self._rtde.start_receiving()

    def get_tcp_pose(self):
        p = self._rtde.receive_data().get("actual_TCP_pose")
        return {"x": p[0], "y": p[1], "z": p[2],
                "rx": p[3], "ry": p[4], "rz": p[5]}


# ---------------------------------------------------------------------------
# DetectorAdapter - YOLO + MC-Dropout UQ.
# ---------------------------------------------------------------------------
class RealDetector:
    """Wraps YOLO + run_uq + uq_analysis. Owns one loaded model per model_id.

    The `detect()` return contract is identical to `SimulatedDetector.detect`:
        (list[DetectionResult], screw_entropy: float | None)
    """

    ENTROPY_THRESHOLD = 0.5  # matches original uq_analysis()

    # Model id space (absolute paths, anchored to the repo so they resolve
    # regardless of CWD). Ids 1 & 2 are the two lighting-calibrated CLOSEUP
    # variants the MAPLE-K loop swaps between (ActiveModel.model_id carries
    # these). Ids 3-5 are the fixed per-task models, selected by detection_type
    # and never touched by the swap logic. Point these at your own weights if
    # different - id 1 in particular is likely your robot-local closeup model
    # (the original used /home/.../pc_all_brightnesses_classified_tests).
    MODEL_PATHS = {
        1: _detect_weights("real_images_with_noscrews_and_screw_fixture"),
        2: _detect_weights("real_images_with_noscrews_and_screw_fixture_extended"),
        3: _detect_weights("real_images_with_noscrews_and_screw_fixture_extended"),
        5: _detect_weights("real_images_with_noscrews_and_screw_fixture"),
        # id 4 (screen segmentation) uses a seg model + a different detect path;
        # see SEG_MODEL_PATHS / detect() below.
    }

    SEG_MODEL_PATHS = {
        4: os.path.join(_DETECTION_MODEL_DIR, "scripts", "runs", "seg",
                        "screen_and_screen_frame.pt"),
    }

    # detection_type string -> base model id (mirrors original change_active_model).
    DETECTION_TYPE_TO_MODEL = {
        "pc_screen": 1,                 # adaptive; runtime id may become 2
        "screw_fixture": 3,
        "topview": 5,
        "screen_or_screen_frame": 4,    # seg model - not yet wired end to end
    }

    # Only the closeup task participates in the MAPLE-K entropy swap loop.
    ADAPTIVE_DETECTION_TYPES = {"pc_screen"}

    # The two closeup variants (base + candidate). For these, screws/noscrews
    # outside a holder are discarded, like the original PC_CLOSEUP path.
    CLOSEUP_MODEL_IDS = frozenset({1, 2})

    # Bbox-detection ids that run through the run_uq / entropy path. The seg
    # model (id 4) is intentionally excluded.
    _BBOX_MODEL_IDS = frozenset(MODEL_PATHS)

    def __init__(self, model_id=1, T=10, uq_config=(0.05,)):
        from ultralytics import YOLO

        self.model_id = model_id
        self.T = T
        self.uq_config = list(uq_config)
        self._models = {}
        # Dedup by path: several ids share the same weights file today, so we
        # only load each file once.
        loaded_by_path = {}
        for mid, path in self.MODEL_PATHS.items():
            if path not in loaded_by_path:
                m = YOLO(path)
                m.fuse()
                loaded_by_path[path] = m
            self._models[mid] = loaded_by_path[path]

    def detect(self, image, model_id):
        # Lazy import so this module works without deepluq installed.
        from inference_uq_single_detection import run_uq

        if model_id not in self._BBOX_MODEL_IDS:
            # Seg model (screen/screen_frame): masks, no UQ, and coordinate
            # extraction via oriented bboxes - none of that is ported onto the
            # core yet. Fail loudly rather than silently mis-detecting.
            raise NotImplementedError(
                f"model_id={model_id} is a segmentation model; the mask/coord "
                "path is not ported onto ScrewDetectionCore yet."
            )

        model = self._models[model_id]
        uq_preds = run_uq(model, image, T=self.T,
                          uq_method="mc_dropout", uq_config=self.uq_config)

        results = []
        for key, det in uq_preds.items():
            if key == "Metrics_Avg":
                continue
            d = det["detection"]
            entropy = d["entropy [classification]"]
            if entropy > self.ENTROPY_THRESHOLD:
                continue  # mirror uq_analysis() filtering

            results.append(DetectionResult(
                label=d["label"],
                box=np.asarray(d["box"], dtype=float),
                score=float(d["score"]),
                mask=d.get("mask"),
                entropy=float(entropy),
            ))

        if model_id in self.CLOSEUP_MODEL_IDS:
            # Closeup task, like the original PC_CLOSEUP path: screws/noscrews
            # come from the UQ pass, holders from a plain confidence-filtered
            # pre-pass, and screw/noscrew detections whose center is outside
            # every holder are discarded.
            holders = self._detect_holders(image, model_id)
            results = [r for r in results
                       if r.label != DetectionClasses.HOLDER.value] + holders
            results = filter_detections_within_holders(results)

        # Entropy of the surviving screw/noscrew detection (post-filter, so a
        # stray detection outside the holder can't feed the entropy window).
        screw_entropy = None
        for r in results:
            if r.label in (DetectionClasses.SCREW.value,
                           DetectionClasses.NOSCREW.value):
                screw_entropy = r.entropy

        return results, screw_entropy

    def _detect_holders(self, image, model_id):
        """Plain (non-UQ) holder pre-pass: predict at conf 0.75 and keep only
        HOLDER boxes. Port of the original's holder detection in
        take_new_image_and_detect()."""
        preds = self._models[model_id].predict(
            image, conf=0.75, verbose=False, iou=0.1
        )
        holders = []
        for box in preds[0].boxes:
            if int(box.cls) != DetectionClasses.HOLDER.value:
                continue
            holders.append(DetectionResult(
                label=DetectionClasses.HOLDER.value,
                box=box.xyxy[0].cpu().numpy().astype(float),
                score=float(box.conf[0]),
                mask=None,
                entropy=None,
            ))
        return holders
