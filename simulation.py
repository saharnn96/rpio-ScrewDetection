"""Offline simulation driver for the screw-detection MAPLE-K loop.

Run with:    python simulation.py

It replaces the three hardware dependencies of the original code with fakes:

  * SimulatedCamera  - synthesizes BGR frames whose BRIGHTNESS changes over
    time (a "lighting change" partway through the run), instead of RealSense.
  * SimulatedRTDE    - returns a fixed TCP pose, instead of the UR robot.
  * SimulatedDetector- returns one holder + one screw detection and an ENTROPY
    derived from frame brightness vs. the model's optimal brightness, instead
    of YOLO + run_uq/deepluq.

How the loop is exercised (the demo "story"):
  - Model 1 is tuned for BRIGHT scenes (optimal brightness ~150).
  - Model 2 is tuned for DIM scenes (optimal brightness ~50).
  - Frames start bright -> model 1 entropy is low -> no anomaly.
  - At `drift_after` ticks the scene goes dim -> model 1 entropy climbs above
    the 0.5 threshold -> Analysis raises an anomaly -> Plan proposes model 2 ->
    Legitimate re-tests model 2 on the recent (now-dim) frames, finds lower
    entropy -> Execute swaps to model 2 -> entropy recovers.

Because the entropy is a deterministic function of the SAVED image's
brightness, Legitimate's disk-based re-evaluation is self-consistent with what
Analysis measured live - exactly how the real LEGITIMATE region re-reads the
rolling-average images from disk.
"""

import json
import logging
import time

import numpy as np

import screwSegmentation_rpio as rpio
from screwSegmentation_rpio import (
    ScrewDetectionCore, DetectionResult, DetectionClasses, build_nodes,
    run_dashboard, _USING_REAL_RPCLPY,
)
from messages import ActionCommand


# ---------------------------------------------------------------------------
# Simulated hardware / model
# ---------------------------------------------------------------------------
class SimulatedCamera:
    """Generates synthetic 1280x720 frames; brightness drops after `drift_after`."""

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

        # Background at the target brightness + a little noise.
        frame = np.full((self.height, self.width, 3), level, dtype=np.uint8)
        noise = np.random.randint(-8, 8, frame.shape, dtype=np.int16)
        frame = np.clip(frame.astype(np.int16) + noise, 0, 255).astype(np.uint8)

        # Draw a "holder" rectangle and a "screw" circle so the saved images
        # look like a plausible detection scene.
        cx, cy = self.width // 2, self.height // 2
        cv2.rectangle(frame, (cx - 200, cy - 150), (cx + 200, cy + 150), (0, 0, 255), 3)
        cv2.circle(frame, (cx, cy), 40, (0, 255, 0), 3)
        return frame


class SimulatedRTDE:
    """Returns a fixed TCP pose; stands in for RTDEReceive."""

    def __init__(self, pose=None):
        self._pose = pose or {"x": 0.4, "y": 0.0, "z": 0.5,
                              "rx": 0.0, "ry": 3.14, "rz": 0.0}

    def get_tcp_pose(self):
        return dict(self._pose)


class SimulatedDetector:
    """Fakes YOLO + run_uq. Entropy = distance(brightness, model optimum)/150.

    Replaces the real `run_uq(...) -> uq_analysis(...)` chain. `model_id` is the
    deployed model; Execute mutates it on a successful swap.
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
            score=0.95, mask=None, entropy=0.05)
        screw = DetectionResult(
            label=DetectionClasses.SCREW.value,
            box=np.array([cx - 40, cy - 40, cx + 40, cy + 40], dtype=float),
            score=max(0.5, 1.0 - entropy), mask=None, entropy=entropy)
        return [holder, screw], entropy


# ---------------------------------------------------------------------------
# Sensor driver - emits SensorData ticks that drive the whole MAPLE-K chain.
# ---------------------------------------------------------------------------
class SensorPublisher(rpio.Node):
    """Stands in for the external sensor/robot that triggers a detection cycle."""

    def __init__(self, config=None, verbose=True):
        super().__init__(config=config, verbose=verbose)
        self._name = "Sensor"

    def emit(self, detection_type, tcp_pose):
        payload = json.dumps({"detection_type": detection_type, "tcp_pose": tcp_pose})
        self.publish_event(event_key="SensorData", message=payload)


def main(num_ticks=24, interval=0.05):
    logging.getLogger().setLevel(logging.INFO)
    print("=" * 78)
    print(f"Screw-detection MAPLE-K simulation  (rpclpy={'REAL' if _USING_REAL_RPCLPY else 'local shim'})")
    print("=" * 78)

    # Build simulated hardware + detector and the shared core.
    camera = SimulatedCamera(drift_after=12)
    rtde = SimulatedRTDE()
    detector = SimulatedDetector(model_id=1)
    core = ScrewDetectionCore(
        camera=camera, rtde=rtde, detector=detector,
        out_dir="./_sim_out", entropy_window_size=8, entropy_threshold=0.5,
        candidate_model_id=2, max_replans=3,
    )
    rpio.set_core(core)

    # Try to load config.yaml (used for real rpclpy); fine if absent offline.
    config = {}
    try:
        import yaml, os
        cfg_path = os.path.join(os.path.dirname(__file__), "config.yaml")
        if os.path.exists(cfg_path):
            with open(cfg_path) as f:
                config = yaml.safe_load(f) or {}
    except Exception as exc:
        print(f"(config.yaml not loaded: {exc})")

    build_nodes(config)
    sensor = SensorPublisher(config.get("Monitor_Config") if config else None)

    # Optional dashboard (no-op on the offline shim).
    run_dashboard(host="127.0.0.1", port=8050, debug=False, start_trust=True)

    print("\nDriving SensorData ticks (lighting goes dim at tick 12)...\n")
    for tick in range(num_ticks):
        print(f"\n---------- tick {tick:02d} ----------")
        sensor.emit(detection_type="pc_screen", tcp_pose=rtde.get_tcp_pose())
        time.sleep(interval)

    # Report the final adapted state.
    cmd = rpio.CORE.detector.model_id
    print("\n" + "=" * 78)
    print(f"Simulation finished. Final deployed model id = {cmd}")
    last_cmd = sensor.read_knowledge(ActionCommand)
    if last_cmd and last_cmd.command:
        print(f"Last action: {last_cmd.command} -> model {last_cmd.model_id} @ {last_cmd.timestamp}")
    print("=" * 78)


if __name__ == "__main__":
    main()
