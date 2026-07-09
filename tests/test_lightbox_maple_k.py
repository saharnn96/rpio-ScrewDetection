"""STEP 3 - Full MAPLE-K loop checks on the light box.

Wires all three layers exactly like `lightbox_main.py` (real camera, mocked
robot) and drives `sensor_data_received` ticks through the real rpclpy/redis
bus (a redis server on 127.0.0.1:6379 is required; checks skip without it):

  1. loop_smoke_real_stack : the real light-box stack (camera + detector as
     installed) completes Monitor -> Analysis cycles and writes knowledge.
  2. loop_forced_model_swap: a scripted detector forces high entropy so the
     whole anomaly -> Plan -> Legitimate -> Execute chain runs and swaps the
     active model, deterministically.
  3. loop_rejected_plan    : the candidate scores WORSE, so Legitimate must
     reject it, exhaust the replan budget and abort without swapping.

Checks 2 and 3 validate the MAPLE-K plumbing itself, so if no camera is
attached they fall back to the SimulatedCamera (clearly announced) instead of
skipping. Check 1 is about the real stack and skips without the camera.

Run on the box:   python tests/test_lightbox_maple_k.py
"""

import os
import tempfile

import numpy as np

from testutil import (
    SkipCheck, run_checks, get_camera, close_camera,
    EventCounter, flush_redis, shutdown_nodes, track_nodes, wait_until,
)

from managed_system import core as ss
from managed_system.core import ScrewDetectionCore, DetectionResult, DetectionClasses
from managed_system.adapters.lightbox import LightboxRTDE, build_lightbox_detector
from managing_system.messages import (
    FrameRef, Detections, EntropyHistory, ActiveModel, ActionCommand,
    InPlanning, ReplanningCounter, LegitResult,
)


# ---------------------------------------------------------------------------
# Wiring helpers.
# ---------------------------------------------------------------------------
class ScriptedDetector:
    """Deterministic detector: entropy depends only on the model id.

    Lets the loop tests force (or forbid) an adaptation regardless of what the
    camera actually sees, while real frames still flow through Monitor and the
    rolling-image window on disk.
    """

    def __init__(self, entropy_by_model, model_id=1):
        self.entropy_by_model = dict(entropy_by_model)
        self.model_id = model_id

    def detect(self, image, model_id):
        h, w = image.shape[:2]
        entropy = self.entropy_by_model.get(model_id, 0.5)
        screw = DetectionResult(
            label=DetectionClasses.SCREW.value,
            box=np.array([w / 2 - 40, h / 2 - 40, w / 2 + 40, h / 2 + 40]),
            score=max(0.05, 1.0 - entropy), mask=None, entropy=entropy,
        )
        return [screw], entropy


def _loop_camera(require_real):
    """Real light-box camera, or (for plumbing checks) the sim camera."""
    try:
        return get_camera(), True
    except SkipCheck:
        if require_real:
            raise
        print("    NOTE: no RealSense found - using SimulatedCamera for this "
              "plumbing check; re-run on the box for the real thing")
        from managed_system.adapters.sim import SimulatedCamera
        return SimulatedCamera(), False


def _wire(camera, detector, window, max_replans=3):
    """Fresh knowledge store + core + the five MAPLE-K nodes on the real
    redis bus; returns a started sensor node."""
    shutdown_nodes()  # previous wiring's subscribers must stop firing
    flush_redis()
    core = ScrewDetectionCore(
        camera=camera, rtde=LightboxRTDE(), detector=detector,
        out_dir=tempfile.mkdtemp(prefix="lightbox_maplek_test_"),
    )
    ss.set_core(core)

    from managing_system.nodes import build_nodes
    from simulation import SensorPublisher, _load_managing_config
    # Real rpclpy nodes need the full per-node wiring config; only the
    # adaptation policy is overridden per test.
    config = _load_managing_config()
    config["Adaptation_Config"] = {
        "entropy_window_size": window, "entropy_threshold": 0.5,
        "candidate_model_id": 2, "max_replans": max_replans,
    }
    track_nodes(build_nodes(config))
    sensor = SensorPublisher(config.get("Monitor_Config"))
    sensor.start()
    track_nodes([sensor])
    return sensor


def _tick(sensor, counter, tick_no, detection_type="pc_screen"):
    """Emit one sensor event and wait for its Monitor->Analysis cycle."""
    sensor.emit(detection_type=detection_type,
                tcp_pose=ss.CORE.rtde.get_tcp_pose())
    assert counter.wait_for("detection_completed", count=tick_no + 1), \
        f"tick {tick_no}: detection_completed did not arrive"


# ---------------------------------------------------------------------------
def loop_smoke_real_stack():
    """Real stack end to end: each tick must produce FrameRef + Detections."""
    camera, _ = _loop_camera(require_real=True)
    detector = build_lightbox_detector(model_id=1, T=3)
    sensor = _wire(camera, detector, window=8)

    counter = EventCounter("detection_completed")

    num_ticks = 3
    for i in range(num_ticks):
        _tick(sensor, counter, i)

    completed = counter.count("detection_completed")
    counter.close()
    assert completed == num_ticks, \
        f"expected {num_ticks} detection_completed events, got {completed}"

    frame_ref = sensor.read_knowledge(FrameRef)
    assert frame_ref is not None and os.path.exists(frame_ref.frame_path), \
        "Monitor did not write a valid FrameRef"
    assert frame_ref.detection_type == "pc_screen"
    assert frame_ref.tcp_pose == ss.CORE.rtde.get_tcp_pose(), \
        "FrameRef must carry the (mocked) TCP pose"

    dets = sensor.read_knowledge(Detections)
    assert dets is not None and isinstance(dets.items, list), \
        "Analysis did not write Detections"

    hist = sensor.read_knowledge(EntropyHistory)
    if dets.screw_entropy is not None:
        assert len(hist.values) > 0, "screw entropy seen but window not updated"
        print(f"    entropy window after {num_ticks} ticks: "
              f"{[f'{v:.3f}' for v in hist.values]}")
    else:
        print("    NOTE: no screw/noscrew detected (empty box?) - entropy "
              "window untouched, which is the correct behaviour")
    print(f"    {num_ticks} full Monitor->Analysis cycles completed")


def loop_forced_model_swap():
    """Anomaly -> Plan -> Legitimate(accept) -> Execute must swap the model."""
    window = 3
    camera, _ = _loop_camera(require_real=False)
    # Active model 1 is 'bad' (0.9), candidate model 2 is 'good' (0.1).
    detector = ScriptedDetector({1: 0.9, 2: 0.1}, model_id=1)
    sensor = _wire(camera, detector, window=window)

    counter = EventCounter("detection_completed", "plan_executed")

    # Window fills with 0.9 entropies; the final tick trips the anomaly and
    # the adaptation chain runs asynchronously on the bus - wait for its end.
    for i in range(window):
        _tick(sensor, counter, i)
    executed = counter.wait_for("plan_executed")
    counter.close()

    assert executed, "plan_executed never fired - adaptation chain did not run"
    assert ss.CORE.detector.model_id == 2, \
        f"detector still on model {ss.CORE.detector.model_id}, expected swap to 2"
    assert sensor.read_knowledge(ActiveModel).model_id == 2

    legit = sensor.read_knowledge(LegitResult)
    assert legit.is_legit and legit.candidate_model_id == 2
    assert legit.candidate_avg_entropy < legit.current_avg_entropy, \
        "Legitimate accepted a candidate that was not better"

    cmd = sensor.read_knowledge(ActionCommand)
    assert cmd.command == "swap_model" and cmd.model_id == 2

    assert not sensor.read_knowledge(InPlanning).in_planning, \
        "InPlanning flag not cleared after Execute"
    assert sensor.read_knowledge(EntropyHistory).values == [], \
        "entropy window not reset for the new model"
    print(f"    swap validated: cand_avg={legit.candidate_avg_entropy:.3f} < "
          f"cur_avg={legit.current_avg_entropy:.3f} -> model 1 -> 2")


def loop_rejected_plan():
    """A WORSE candidate must be rejected until max replans, with no swap."""
    window, max_replans = 3, 2
    camera, _ = _loop_camera(require_real=False)
    # Candidate model 2 is even worse than the active model 1.
    detector = ScriptedDetector({1: 0.9, 2: 0.95}, model_id=1)
    sensor = _wire(camera, detector, window=window, max_replans=max_replans)

    counter = EventCounter("detection_completed", "plan_executed",
                           "plan_rejected")

    for i in range(window):
        _tick(sensor, counter, i)

    # The reject/re-plan ping-pong runs asynchronously; the abort (budget
    # exhausted) is knowledge-only, so wait on the InPlanning flag clearing.
    assert counter.wait_for("plan_rejected", count=max_replans), \
        "plan_rejected never reached the replan budget"
    assert wait_until(
        lambda: not sensor.read_knowledge(InPlanning).in_planning), \
        "adaptation was never aborted (InPlanning still set)"
    rejected = counter.count("plan_rejected")
    executed = counter.count("plan_executed")
    counter.close()

    assert not executed, "a worse candidate must never reach Execute"
    assert rejected == max_replans, \
        f"expected {max_replans} plan_rejected events, got {rejected}"
    assert ss.CORE.detector.model_id == 1, "model must NOT be swapped"
    assert not sensor.read_knowledge(LegitResult).is_legit
    assert sensor.read_knowledge(ReplanningCounter).count == max_replans + 1, \
        "replan budget not exhausted as expected"
    assert not sensor.read_knowledge(InPlanning).in_planning, \
        "InPlanning must be cleared after the adaptation is aborted"
    print(f"    candidate rejected {rejected + 1}x, budget exhausted, "
          "model 1 kept - Legitimate gate works")


if __name__ == "__main__":
    try:
        run_checks(
            [
                loop_smoke_real_stack,
                loop_forced_model_swap,
                loop_rejected_plan,
            ],
            "STEP 3: light-box MAPLE-K loop checks",
        )
    finally:
        shutdown_nodes()
        close_camera()
