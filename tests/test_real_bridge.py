"""STEP 4 - pendant contract (XML-RPC bridge) checks.

Exercises the exact production wiring of `bridge.py` (used by `real_main.py`)
- RobotBridge + LoggingXMLRPCServer built by `build_server()` - over a REAL XML-RPC
client connection, the same way the UR pendant calls it. Only the hardware
adapters are replaced (scripted camera/detector, sim RTDE), so these checks
run on any machine:

  1. rpc_surface_matches_original : the ten registered method names are the
     ones the original python/screwSegmentation.py served on port 50000.
  2. rpc_take_and_detect          : take_new_image_and_detect blocks until the
     MAPLE-K cycle wrote Detections, returns None, FrameRef carries the pose.
  3. rpc_coords_label_routing     : screw / noscrew / holder coords come back
     in the base frame with the RIGHT labels (regression guard for the
     SCREW<->NOSCREW enum swap) and the right geometry.
  4. rpc_aux_surface              : calibration translation, exposure
     roundtrip, and the no-op display/timer RPCs.
  5. rpc_fault_correlation        : errors surface as Faults with a
     [rpc:xxxxxxxx] correlation id that is also in xmlrpc_errors.log, and the
     server survives to serve the next call.
  6. rpc_stale_coords_cleared     : a cycle with no detections must not serve
     the previous cycle's coordinates.

Run anywhere:   python tests/test_real_bridge.py
(Needs a redis server on 127.0.0.1:6379 for the rpclpy bus; skips without it.)
"""

import os
import re
import tempfile
import threading
import xmlrpc.client

import numpy as np

from testutil import SkipCheck, run_checks, flush_redis, track_nodes, shutdown_nodes

from managed_system import core as ss
from managed_system.core import ScrewDetectionCore, DetectionResult, DetectionClasses
from managed_system.adapters.sim import SimulatedRTDE

# The synchronous UR pendant surface (original screwSegmentation.py, port 50000).
EXPECTED_RPC_SURFACE = {
    "get_trans_T_cam2gripper",
    "take_new_image_and_detect", "get_detected_object_coords",
    "get_exposure_time", "set_exposure_time",
    "enable_show_detections", "disable_show_detections",
    "start_timer", "switch_process", "stop_timer",
}

POSE_ZERO = {"x": 0.0, "y": 0.0, "z": 0.0, "rx": 0.0, "ry": 0.0, "rz": 0.0}

# Camera geometry the coordinate expectations below are derived from.
FX = FY = 600.0
PPX, PPY = 640.0, 360.0
DEPTH_UNITS = 0.001
CLOSE_DEPTH = 165  # img_point_to_cam_point forces this for view="close"
PIXEL_OFF = 120    # screw is +120 px right of center, noscrew -120 px

Z_CLOSE = CLOSE_DEPTH * DEPTH_UNITS                  # 0.165 m
X_OFF = PIXEL_OFF * Z_CLOSE / FX                     # 0.033 m


class ScriptedCamera:
    """Synthetic camera with the same depth/exposure surface as RealCamera."""

    def __init__(self, width=1280, height=720):
        self.width, self.height = width, height
        self.exposure_us = 2500

    def get_color_image(self):
        return np.full((self.height, self.width, 3), 128, dtype=np.uint8)

    def get_depth_snapshot(self):
        return {
            "depth_image": np.zeros((self.height, self.width), dtype=np.uint16),
            "fx": FX, "fy": FY, "ppx": PPX, "ppy": PPY,
            "depth_units": DEPTH_UNITS,
        }

    def get_exposure(self):
        return self.exposure_us

    def set_exposure(self, exposure_us):
        self.exposure_us = exposure_us


class ScriptedBridgeDetector:
    """One holder + one screw (+120px) + one noscrew (-120px), or nothing.

    Distinct, known positions so the coords checks can tell screw and noscrew
    apart - a label mix-up produces the wrong sign on x and fails loudly.
    """

    def __init__(self):
        self.model_id = 1
        self.mode = "objects"  # "objects" | "empty" | "boom"

    def detect(self, image, model_id):
        if self.mode == "boom":
            raise RuntimeError("boom (scripted detector failure)")
        if self.mode == "empty":
            return [], None

        def det(label, cx_px):
            return DetectionResult(
                label=label,
                box=np.array([cx_px - 30, PPY - 30, cx_px + 30, PPY + 30],
                             dtype=float),
                score=0.9, mask=None, entropy=0.1,
            )

        holder = DetectionResult(
            label=DetectionClasses.HOLDER.value,
            box=np.array([PPX - 200, PPY - 80, PPX + 200, PPY + 80], dtype=float),
            score=0.95, mask=None, entropy=0.05,
        )
        screw = det(DetectionClasses.SCREW.value, PPX + PIXEL_OFF)
        noscrew = det(DetectionClasses.NOSCREW.value, PPX - PIXEL_OFF)
        return [holder, screw, noscrew], 0.1


# ---------------------------------------------------------------------------
# One production-wired bridge + server + client, shared by all checks.
# ---------------------------------------------------------------------------
_ctx = None


def _identity_calib_folder():
    """CalibData dir with an identity hand-eye transform: with a zero TCP pose,
    base-frame coords == camera-frame coords, so expectations are exact."""
    folder = tempfile.mkdtemp(prefix="real_bridge_calib_")
    ft = os.path.join(folder, "FinalTransforms")
    os.makedirs(ft)
    np.savez(os.path.join(ft, "T_cam2gripper_Method_1.npz"), np.eye(4))
    return folder


def _get_ctx():
    """Wire core + MAPLE-K nodes + RobotBridge + XMLRPC server (once)."""
    global _ctx
    if _ctx is not None:
        return _ctx
    flush_redis()  # raises SkipCheck when no redis server is reachable

    from managing_system.nodes import build_nodes
    from simulation import _load_managing_config
    from bridge import RobotBridge, build_server

    detector = ScriptedBridgeDetector()
    core = ScrewDetectionCore(
        camera=ScriptedCamera(), rtde=SimulatedRTDE(), detector=detector,
        out_dir=tempfile.mkdtemp(prefix="real_bridge_test_"),
        calib_folder=_identity_calib_folder(),
    )
    ss.set_core(core)

    # Real rpclpy nodes need the full per-node wiring config from config.yaml.
    config = _load_managing_config()
    track_nodes(build_nodes(config))
    bridge = RobotBridge(config.get("RobotBridge_Config"))
    # A failed cycle only surfaces to the pendant as a timeout (see
    # rpc_fault_correlation); keep that wait short. Healthy cycles take <1s.
    bridge.CYCLE_TIMEOUT_S = 10
    bridge.register_callbacks()
    bridge.start()
    track_nodes([bridge])

    server = build_server(bridge, xmlrpc_port=0, host="127.0.0.1")
    port = server.server_address[1]
    threading.Thread(target=server.serve_forever, daemon=True).start()

    client = xmlrpc.client.ServerProxy(f"http://127.0.0.1:{port}",
                                       allow_none=True)
    _ctx = {"client": client, "server": server, "bridge": bridge,
            "detector": detector}
    return _ctx


def _shutdown():
    if _ctx is not None:
        # Close the client's keep-alive connection FIRST: the single-threaded
        # server sits in handle() on the persistent HTTP/1.1 connection, and
        # server.shutdown() blocks until it returns to the accept loop.
        _ctx["client"]("close")()
        _ctx["server"].shutdown()
        _ctx["server"].server_close()
    shutdown_nodes()


# ---------------------------------------------------------------------------
def rpc_surface_matches_original():
    """Registered method names == the original pendant surface, exactly."""
    ctx = _get_ctx()
    registered = set(ctx["server"].funcs)
    assert registered == EXPECTED_RPC_SURFACE, (
        f"missing: {EXPECTED_RPC_SURFACE - registered}, "
        f"unexpected: {registered - EXPECTED_RPC_SURFACE}"
    )


def rpc_take_and_detect():
    """take_new_image_and_detect completes a MAPLE-K cycle and returns None."""
    from managing_system.messages import FrameRef, Detections
    ctx = _get_ctx()
    ctx["detector"].mode = "objects"

    ret = ctx["client"].take_new_image_and_detect(POSE_ZERO, "pc_screen")
    assert ret is None, f"expected None (coords come via a second RPC), got {ret!r}"

    frame_ref = ctx["bridge"].read_knowledge(FrameRef)
    assert frame_ref is not None and os.path.exists(frame_ref.frame_path), \
        "Monitor did not write a valid FrameRef"
    assert frame_ref.tcp_pose == POSE_ZERO, \
        f"FrameRef must carry the pendant's pose, got {frame_ref.tcp_pose}"

    dets = ctx["bridge"].read_knowledge(Detections)
    labels = sorted(d["label"] for d in dets.items)
    assert labels == [DetectionClasses.HOLDER.value,
                      DetectionClasses.NOSCREW.value,
                      DetectionClasses.SCREW.value], f"bad labels: {labels}"
    print(f"    cycle completed, {len(dets.items)} detections in knowledge")


def rpc_coords_label_routing():
    """screw/noscrew/holder each map to THEIR detection's base-frame coords.

    Identity calibration + zero pose + view='close' makes the expected values
    exact: x = px_offset * 0.165 / fx, y = 0, z = 0.165. A SCREW<->NOSCREW
    label swap flips the sign of x and fails this check.
    """
    ctx = _get_ctx()
    ctx["detector"].mode = "objects"
    ctx["client"].take_new_image_and_detect(POSE_ZERO, "pc_screen")

    def coords(obj_type):
        got = ctx["client"].get_detected_object_coords(obj_type, "close")
        assert len(got) % 3 == 0, f"{obj_type}: not a flat [x,y,z,...]: {got}"
        return got

    for obj_type, expected in [
        ("screw", [X_OFF, 0.0, Z_CLOSE]),
        ("noscrew", [-X_OFF, 0.0, Z_CLOSE]),
        ("holder", [0.0, 0.0, Z_CLOSE]),
    ]:
        got = coords(obj_type)
        assert len(got) == 3, f"{obj_type}: expected 1 object, got {got}"
        assert all(abs(g - e) < 1e-9 for g, e in zip(got, expected)), \
            f"{obj_type}: expected {expected}, got {got}"
    print(f"    screw at x=+{X_OFF:.3f}, noscrew at x=-{X_OFF:.3f}, "
          f"holder at x=0 - labels route correctly")


def rpc_aux_surface():
    """Calibration translation, exposure roundtrip, and the no-op RPCs."""
    ctx = _get_ctx()
    client = ctx["client"]

    assert client.get_trans_T_cam2gripper() == [0.0, 0.0, 0.0], \
        "identity calibration must give a zero translation"

    assert client.get_exposure_time() == 2500
    client.set_exposure_time(400)
    assert client.get_exposure_time() == 400
    client.set_exposure_time(2500)  # restore

    for name in ("enable_show_detections", "disable_show_detections",
                 "stop_timer"):
        assert getattr(client, name)() is None, f"{name} must return None"
    assert client.start_timer(["p1", "p2"]) is None
    assert client.switch_process("p1") is None


def rpc_fault_correlation():
    """Errors become Faults with a correlation id that is in the log file,
    and the server keeps serving afterwards."""
    from bridge import _RPC_LOG_FILENAME
    ctx = _get_ctx()
    client = ctx["client"]

    def expect_fault(fn, *want_substrings):
        try:
            fn()
        except xmlrpc.client.Fault as fault:
            for want in want_substrings:
                assert want in fault.faultString, \
                    f"fault missing {want!r}: {fault.faultString}"
            return fault.faultString
        raise AssertionError("expected an xmlrpc Fault, got a normal return")

    # Invalid obj_type -> ValueError; unported screen type -> NotImplementedError.
    msg = expect_fault(lambda: client.get_detected_object_coords("bogus"),
                       "[rpc:", "ValueError")
    expect_fault(lambda: client.get_detected_object_coords("screen"),
                 "[rpc:", "NotImplementedError")

    # A failure deep inside the MAPLE-K cycle must also surface as a Fault
    # (not a hang forever). On the real async bus the exception cannot travel
    # back to the pendant's synchronous call: it is logged by the Analysis
    # node (which must survive it - see nodes.safe_callback) and the bridge
    # times the cycle out.
    ctx["detector"].mode = "boom"
    expect_fault(
        lambda: client.take_new_image_and_detect(POSE_ZERO, "pc_screen"),
        "[rpc:", "TimeoutError")
    ctx["detector"].mode = "objects"

    # The correlation id from the Fault must be greppable in the log file.
    corr_id = re.search(r"\[rpc:([0-9a-f]{8})\]", msg).group(1)
    assert os.path.exists(_RPC_LOG_FILENAME), f"{_RPC_LOG_FILENAME} not written"
    with open(_RPC_LOG_FILENAME, encoding="utf-8", errors="replace") as f:
        assert corr_id in f.read(), \
            f"correlation id {corr_id} not found in {_RPC_LOG_FILENAME}"

    # Server must still be alive.
    assert client.get_exposure_time() == 2500
    print(f"    faults carry [rpc:{corr_id}], logged to "
          f"{os.path.basename(_RPC_LOG_FILENAME)}, server survived")


def rpc_stale_coords_cleared():
    """A cycle with zero detections must not serve the previous cycle's coords."""
    ctx = _get_ctx()
    client = ctx["client"]

    ctx["detector"].mode = "objects"
    client.take_new_image_and_detect(POSE_ZERO, "pc_screen")
    assert client.get_detected_object_coords("screw", "close"), \
        "setup failed: first cycle should have detections"

    ctx["detector"].mode = "empty"
    client.take_new_image_and_detect(POSE_ZERO, "pc_screen")
    got = client.get_detected_object_coords("screw", "close")
    assert got == [], f"stale coords served after an empty cycle: {got}"
    ctx["detector"].mode = "objects"
    print("    empty cycle correctly returns [] instead of stale coords")


if __name__ == "__main__":
    try:
        run_checks(
            [
                rpc_surface_matches_original,
                rpc_take_and_detect,
                rpc_coords_label_routing,
                rpc_aux_surface,
                rpc_fault_correlation,
                rpc_stale_coords_cleared,
            ],
            "STEP 4: real_main pendant contract (XML-RPC bridge) checks",
        )
    finally:
        _shutdown()
