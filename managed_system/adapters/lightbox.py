"""Layer 1 (Hardware layer) - LIGHTBOX adapters for the educational light-box setup.

The light box has a REAL RealSense camera but NO robot yet, so this module
mixes one real adapter with one mock:

  * LightboxCamera  - the real pyrealsense2 pipeline (reuses RealCamera from
                      `real.py` unchanged) plus a `close()` helper so
                      test scripts can release the device.
  * LightboxRTDE    - mocked robot telemetry. Same interface as RealRTDE
                      (`get_tcp_pose()`), returns a fixed-but-settable pose.
                      There is no outgoing "move robot" adapter in this
                      architecture - robot commands arrive FROM the UR pendant
                      via XMLRPC (see real_main.py) - so mocking telemetry is
                      all that is needed until the robot arrives.
  * build_lightbox_detector() - the real YOLO+UQ detector if the ML stack
                      (ultralytics/torch/deepluq) is installed, otherwise the
                      SimulatedDetector with a loud warning. On the box you
                      want the real one: `pip install ultralytics torch
                      torchmetrics deepluq pyrealsense2`.

Wire-up entry point: `lightbox_main.py`. Test scripts: `tests/`.
"""

from managed_system.adapters.real import RealCamera


# ---------------------------------------------------------------------------
# CameraAdapter - REAL RealSense camera (identical behaviour to deployment).
# ---------------------------------------------------------------------------
class LightboxCamera(RealCamera):
    """Real RealSense camera for the light box.

    Behaviour is intentionally identical to `RealCamera` - the point of the
    box is to exercise the exact camera path that will run in the field.
    Only adds `close()` so tests can start/stop the pipeline repeatedly.
    """

    def close(self):
        try:
            self.pipeline.stop()
        except Exception:
            pass  # already stopped / never started


# ---------------------------------------------------------------------------
# RTDEAdapter - MOCKED robot telemetry (no robot on the box yet).
# ---------------------------------------------------------------------------
class LightboxRTDE:
    """Stands in for RealRTDE until the box gets a robot.

    Returns a constant TCP pose like SimulatedRTDE, but the pose can be
    updated at runtime (`set_tcp_pose`) so you can mimic "the robot moved
    here" while driving the box by hand.
    """

    DEFAULT_POSE = {"x": 0.4, "y": 0.0, "z": 0.5,
                    "rx": 0.0, "ry": 3.14, "rz": 0.0}

    def __init__(self, pose=None):
        self._pose = dict(pose) if pose else dict(self.DEFAULT_POSE)

    def get_tcp_pose(self):
        return dict(self._pose)

    def set_tcp_pose(self, **components):
        """Update pose components, e.g. set_tcp_pose(z=0.3, rx=0.1)."""
        unknown = set(components) - set(self.DEFAULT_POSE)
        if unknown:
            raise ValueError(f"Unknown pose components: {sorted(unknown)}; "
                             f"valid: {sorted(self.DEFAULT_POSE)}")
        self._pose.update({k: float(v) for k, v in components.items()})


# ---------------------------------------------------------------------------
# DetectorAdapter - real YOLO+UQ when installed, sim fallback otherwise.
# ---------------------------------------------------------------------------
def build_lightbox_detector(model_id=1, allow_sim_fallback=True, **real_kwargs):
    """Return the best detector available on this machine.

    Tries RealDetector (YOLO + MC-dropout UQ, needs ultralytics/torch/deepluq
    and the weight files under screw_detection/detection_model). If that
    fails and `allow_sim_fallback` is True, returns SimulatedDetector so the
    rest of the stack can still be exercised.

    `real_kwargs` are forwarded to RealDetector (e.g. T=3 to speed up the
    MC-dropout passes during testing).
    """
    try:
        from managed_system.adapters.real import RealDetector
        return RealDetector(model_id=model_id, **real_kwargs)
    except Exception as exc:
        if not allow_sim_fallback:
            raise
        print("=" * 70)
        print(f"WARNING: real detector unavailable ({type(exc).__name__}: {exc})")
        print("Falling back to SimulatedDetector - detections are FAKE.")
        print("For real detection on the box: pip install ultralytics torch "
              "torchmetrics deepluq")
        print("=" * 70)
        from managed_system.adapters.sim import SimulatedDetector
        return SimulatedDetector(model_id=model_id)
