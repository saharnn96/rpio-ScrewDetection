"""SIMULATED adapters (managed system, Layer 1).

Each class satisfies the adapter interface documented in `managed_system/core.py`.
Nothing above Layer 1 cares whether it is a sim or a real adapter -
`simulation.py` picks these; `real_main.py` picks the ones in `real.py`.
"""

import numpy as np

from managed_system.core import DetectionResult, DetectionClasses


# ---------------------------------------------------------------------------
# CameraAdapter - synthesizes a BGR frame whose brightness drops mid-run.
# ---------------------------------------------------------------------------
class SimulatedCamera:
    """Stands in for the RealSense pipeline.

    The frame is uniform grey at `bright_level` for the first `drift_after`
    calls, then at `dim_level`. That brightness swing is what drives the whole
    MAPLE-K adaptation via `SimulatedDetector`.
    """

    def __init__(self, width=1280, height=720, bright_level=150, dim_level=45,
                 drift_after=12):
        self.width = width
        self.height = height
        self.bright_level = bright_level
        self.dim_level = dim_level
        self.drift_after = drift_after
        self._tick = 0

    def get_color_image(self):
        import cv2
        level = self.bright_level if self._tick < self.drift_after else self.dim_level
        self._tick += 1

        frame = np.full((self.height, self.width, 3), level, dtype=np.uint8)
        noise = np.random.randint(-8, 8, frame.shape, dtype=np.int16)
        frame = np.clip(frame.astype(np.int16) + noise, 0, 255).astype(np.uint8)

        cx, cy = self.width // 2, self.height // 2
        cv2.rectangle(frame, (cx - 200, cy - 150), (cx + 200, cy + 150), (0, 0, 255), 3)
        cv2.circle(frame, (cx, cy), 40, (0, 255, 0), 3)
        return frame


# ---------------------------------------------------------------------------
# RTDEAdapter - fixed pose; robot kinematics aren't part of the loop.
# ---------------------------------------------------------------------------
class SimulatedRTDE:
    """Stands in for RTDEReceive. Returns a constant TCP pose."""

    def __init__(self, pose=None):
        self._pose = pose or {"x": 0.4, "y": 0.0, "z": 0.5,
                              "rx": 0.0, "ry": 3.14, "rz": 0.0}

    def get_tcp_pose(self):
        return dict(self._pose)


# ---------------------------------------------------------------------------
# DetectorAdapter - fake YOLO + UQ whose entropy is derived from brightness.
# ---------------------------------------------------------------------------
class SimulatedDetector:
    """Stands in for YOLO + run_uq + uq_analysis.

    entropy = clip(|mean_brightness - optimum[model_id]| / 150, 0.02, 0.95)

    Model 1 is calibrated for bright scenes, Model 2 for dim scenes. Because
    entropy is a pure function of the image, Legitimate's disk-based re-eval
    is guaranteed to be consistent with what Analyze measured live - exactly
    the property the real LEGITIMATE region relies on.
    """

    OPTIMAL_BRIGHTNESS = {1: 150.0, 2: 50.0}

    def __init__(self, model_id=1):
        self.model_id = model_id

    def _entropy(self, image, model_id):
        import cv2
        brightness = float(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).mean())
        optimum = self.OPTIMAL_BRIGHTNESS.get(model_id, 150.0)
        return float(np.clip(abs(brightness - optimum) / 150.0, 0.02, 0.95))

    def detect(self, image, model_id):
        """Return (list[DetectionResult], screw_entropy) for one holder + one screw."""
        if image is None:
            return [], None
        h, w = image.shape[:2]
        cx, cy = w // 2, h // 2
        entropy = self._entropy(image, model_id)

        holder = DetectionResult(
            label=DetectionClasses.HOLDER.value,
            box=np.array([cx - 200, cy - 150, cx + 200, cy + 150], dtype=float),
            score=0.95, mask=None, entropy=0.05,
        )
        screw = DetectionResult(
            label=DetectionClasses.SCREW.value,
            box=np.array([cx - 40, cy - 40, cx + 40, cy + 40], dtype=float),
            score=max(0.5, 1.0 - entropy), mask=None, entropy=entropy,
        )
        return [holder, screw], entropy
