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
  * Fill in the calibration folder used by the coord-transform helpers if you
    port them onto the core.
  * The interfaces here must match those documented in `screwSegmentation.py`.
"""

import numpy as np

from screwSegmentation import DetectionResult, DetectionClasses


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
        sensor = self.pipeline.get_active_profile().get_device().query_sensors()[1]
        sensor.set_option(rs.option.exposure, exposure_us)

        self.last_depth_frame = None  # kept for optional coord-transform port

    def get_color_image(self):
        frames = self.align.process(self.pipeline.wait_for_frames(5000))
        color = frames.get_color_frame()
        self.last_depth_frame = frames.get_depth_frame()
        return np.asanyarray(color.get_data())


# ---------------------------------------------------------------------------
# RTDEAdapter - UR robot telemetry.
# ---------------------------------------------------------------------------
class RealRTDE:
    """Wraps RTDEReceive. The original code streams tool_current and
    runtime_state too; only the TCP pose is required by the loop."""

    def __init__(self, robot_ip):
        # Import path mirrors what the original screwSegmentation.py uses.
        from RDTEReceive.RTDEReceive import RTDEReceive

        self._rtde = RTDEReceive(robot_ip)
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

    # TODO: point these at the real .pt files used in your setup.
    MODEL_PATHS = {
        1: "../detection_model/scripts/runs/detect/real_images_with_noscrews_and_screw_fixture/weights/best.pt",
        2: "../detection_model/scripts/runs/detect/real_images_with_noscrews_and_screw_fixture_extended/weights/best.pt",
    }

    def __init__(self, model_id=1, T=10, uq_config=(0,)):
        from ultralytics import YOLO

        self.model_id = model_id
        self.T = T
        self.uq_config = list(uq_config)
        self._models = {}
        for mid, path in self.MODEL_PATHS.items():
            m = YOLO(path)
            m.fuse()
            self._models[mid] = m

    def detect(self, image, model_id):
        # Lazy import so this module works without deepluq installed.
        from inference_uq_single_detection import run_uq

        model = self._models[model_id]
        uq_preds = run_uq(model, image, T=self.T,
                          uq_method="mc_dropout", uq_config=self.uq_config)

        results = []
        screw_entropy = None
        for key, det in uq_preds.items():
            if key == "Metrics_Avg":
                continue
            d = det["detection"]
            entropy = d["entropy [classification]"]
            if entropy > self.ENTROPY_THRESHOLD:
                continue  # mirror uq_analysis() filtering

            dr = DetectionResult(
                label=d["label"],
                box=np.asarray(d["box"], dtype=float),
                score=float(d["score"]),
                mask=d.get("mask"),
                entropy=float(entropy),
            )
            results.append(dr)
            if dr.label in (DetectionClasses.SCREW.value,
                            DetectionClasses.NOSCREW.value):
                screw_entropy = dr.entropy

        return results, screw_entropy
