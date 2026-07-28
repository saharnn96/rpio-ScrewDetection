"""STEP 5 - REAL adapter (field deployment) checks.

Validates `managed_system/adapters/real.py` + the deployment prerequisites of
`real_main.py`.
The configuration/parity checks run on any machine; the hardware checks skip
cleanly when the camera / robot / ML stack is not present, so on the box the
same script verifies the real thing:

  config (run anywhere)
    1. check_detection_class_order   : enum values match the trained models
                                       (regression guard for the label swap).
    2. check_detector_static_config  : uq_config=0.05, closeup ids, routing
                                       tables - parity with the original.
    3. check_model_weights_on_disk   : every MODEL_PATHS/SEG_MODEL_PATHS weight
                                       file exists.
    4. check_original_modules_importable : the vendored RDTEReceive package
                                       and managed_system.uq resolve from
                                       the repo root.
    5. check_calibration_data        : the hand-eye npz real_main loads by
                                       default is a valid rigid transform.

  hardware (skip without it)
    6. check_real_detector_class_names   : model.names of the deployed weights
                                           agree with DetectionClasses.
    7. check_real_detector_contract     : detect() honours the adapter
                                           contract + the inside-holder filter.
    8. check_real_detector_seg_model    : seg model id (4) returns oriented
                                           bboxes, no UQ/entropy.
    9. check_real_camera                : RealCamera color/depth/exposure.
   10. check_real_rtde                  : RealRTDE TCP pose from the robot
                                           (ROBOT_IP env var, default
                                           192.168.1.100).

Run on the box:   python tests/test_real_adapters.py
"""

import importlib.util
import inspect
import os
import socket
import tempfile

import numpy as np

from testutil import SkipCheck, run_checks

ROBOT_IP = os.environ.get("ROBOT_IP", "192.168.1.100")
RTDE_PORT = 30004

_detector = None


def _require_module(name, hint):
    if importlib.util.find_spec(name) is None:
        raise SkipCheck(f"{name} not installed - {hint}")


def _get_real_detector():
    """Load RealDetector (and its weights) once for all detector checks."""
    global _detector
    _require_module("ultralytics",
                    "pip install ultralytics torch on the box machine")
    if _detector is None:
        from managed_system.adapters.real import RealDetector
        _detector = RealDetector(model_id=1, T=3)
    return _detector


def _require_uq_stack():
    """detect() needs run_uq, which pulls in torch/deepluq/torchmetrics."""
    try:
        import managed_system.uq  # noqa: F401
    except ImportError as exc:
        raise SkipCheck(f"UQ stack unavailable ({exc}) - "
                        "pip install torch torchmetrics deepluq on the box")


# ---------------------------------------------------------------------------
# Configuration / parity checks - run on any machine.
# ---------------------------------------------------------------------------
def check_detection_class_order():
    """Label ids MUST stay in the trained models' class order (0=holder,
    1=noscrew, 2=screw) or get_detected_object_coords sends the robot to the
    wrong holes. Guard against the SCREW<->NOSCREW swap regression."""
    from managed_system.core import DetectionClasses
    assert DetectionClasses.HOLDER.value == 0
    assert DetectionClasses.NOSCREW.value == 1
    assert DetectionClasses.SCREW.value == 2


def check_detector_static_config():
    """RealDetector settings that must stay in parity with the original."""
    from managed_system.adapters.real import RealDetector

    params = inspect.signature(RealDetector.__init__).parameters
    assert params["uq_config"].default == (0.05,), \
        "uq_config must default to 0.05 dropout (original run_uq call)"
    assert params["T"].default == 10, "T must default to 10 forward passes"

    assert RealDetector.ENTROPY_THRESHOLD == 0.5
    assert RealDetector.CLOSEUP_MODEL_IDS == {1, 2}, \
        "the two closeup variants are the only holder-filtered models"
    assert RealDetector.ADAPTIVE_DETECTION_TYPES == {"pc_screen"}
    assert set(RealDetector.DETECTION_TYPE_TO_MODEL) == {
        "pc_screen", "screw_fixture", "topview", "screen_or_screen_frame"
    }, "detection_type routing must cover the four pendant strings"
    assert (RealDetector.DETECTION_TYPE_TO_MODEL["pc_screen"]
            in RealDetector.CLOSEUP_MODEL_IDS), \
        "pc_screen must start on a closeup (adaptive) model"


def check_model_weights_on_disk():
    """Every bbox + seg model id must point at an existing weights file."""
    from managed_system.adapters.real import RealDetector
    all_paths = {**RealDetector.MODEL_PATHS, **RealDetector.SEG_MODEL_PATHS}
    missing = [f"id {mid}: {path}"
               for mid, path in all_paths.items()
               if not os.path.exists(path)]
    assert not missing, "missing weight files:\n      " + "\n      ".join(missing)
    print(f"    {len(all_paths)} weight paths verified")


def check_original_modules_importable():
    """RealRTDE and detect() lazily import vendored modules that live under
    screw_detection/python - importing real_adapters must make them resolvable
    from any CWD."""
    import managed_system.adapters.real  # noqa: F401  (installs the sys.path entry)
    for mod in ("RDTEReceive", "managed_system.uq"):
        assert importlib.util.find_spec(mod) is not None, \
            f"{mod} not importable - sys.path shim in adapters/real.py broken?"


def check_calibration_data():
    """The default hand-eye calibration must load and be a rigid transform."""
    from real_main import DEFAULT_CALIB_FOLDER
    from managed_system.core import ScrewDetectionCore

    path = os.path.join(DEFAULT_CALIB_FOLDER, "FinalTransforms",
                        "T_cam2gripper_Method_1.npz")
    assert os.path.exists(path), f"calibration file missing: {path}"

    T = np.load(path)["arr_0"]
    assert T.shape == (4, 4), f"expected 4x4 transform, got {T.shape}"
    R = T[:3, :3]
    assert np.allclose(R @ R.T, np.eye(3), atol=1e-5), "rotation not orthonormal"
    assert abs(np.linalg.det(R) - 1.0) < 1e-5, "rotation determinant != 1"
    assert np.allclose(T[3], [0, 0, 0, 1]), "bottom row must be [0,0,0,1]"

    core = ScrewDetectionCore(
        camera=None, rtde=None, detector=None,
        out_dir=tempfile.mkdtemp(prefix="real_calib_test_"),
        calib_folder=DEFAULT_CALIB_FOLDER,
    )
    t = core.get_translation_T_cam2gripper()
    assert len(t) == 3 and all(isinstance(x, float) for x in t)
    assert all(abs(x) < 0.5 for x in t), \
        f"cam-to-gripper translation implausibly large: {t} (meters expected)"
    print(f"    T_cam2gripper OK, translation = "
          f"[{t[0]:.4f}, {t[1]:.4f}, {t[2]:.4f}] m")


# ---------------------------------------------------------------------------
# Hardware checks - skip when the stack / device is not present.
# ---------------------------------------------------------------------------
def check_real_detector_class_names():
    """The deployed weights' own class names must agree with DetectionClasses.
    This is the definitive label-order guard against the actual model files."""
    from managed_system.core import DetectionClasses
    det = _get_real_detector()

    for mid in sorted(det.CLOSEUP_MODEL_IDS):
        names = {int(k): str(v).lower() for k, v in det._models[mid].names.items()}
        for cls in DetectionClasses:
            assert names.get(cls.value) == cls.name.lower(), (
                f"model id {mid}: class {cls.value} is "
                f"{names.get(cls.value)!r} in the weights but "
                f"{cls.name.lower()!r} in DetectionClasses - label mismatch! "
                f"(full names: {names})"
            )
        print(f"    model id {mid}: names {names} match DetectionClasses")


def check_real_detector_contract():
    """detect() contract + the inside-holder filter postcondition."""
    from managed_system.core import DetectionResult, DetectionClasses
    det = _get_real_detector()
    _require_uq_stack()

    image = np.full((720, 1280, 3), 128, dtype=np.uint8)
    results, screw_entropy = det.detect(image, model_id=1)

    assert isinstance(results, list)
    assert screw_entropy is None or 0.0 <= screw_entropy <= det.ENTROPY_THRESHOLD, \
        f"screw_entropy {screw_entropy} above the filter threshold leaked through"

    holders = [r for r in results if r.label == DetectionClasses.HOLDER.value]
    for r in results:
        assert isinstance(r, DetectionResult)
        if r.label in (DetectionClasses.SCREW.value,
                       DetectionClasses.NOSCREW.value):
            cx = (r.box[0] + r.box[2]) / 2
            cy = (r.box[1] + r.box[3]) / 2
            assert any(h.box[0] <= cx <= h.box[2] and h.box[1] <= cy <= h.box[3]
                       for h in holders), \
                f"screw/noscrew at ({cx:.0f},{cy:.0f}) outside every holder - " \
                "inside-holder filter not applied"
    print(f"    {len(results)} detection(s) on a synthetic frame, "
          f"screw_entropy={screw_entropy} - contract + filter hold")


def check_real_detector_seg_model():
    """model_id=4 (screen/screen_frame) returns oriented bboxes, no UQ."""
    from managed_system.core import DetectionResult
    det = _get_real_detector()

    image = np.full((720, 1280, 3), 128, dtype=np.uint8)
    results, screw_entropy = det.detect(image, model_id=4)

    assert isinstance(results, list)
    assert screw_entropy is None, \
        "seg model is never adaptive; detect() must not compute an entropy"
    for r in results:
        assert isinstance(r, DetectionResult)
        assert r.entropy is None
    assert isinstance(det.last_oriented_bbox_screen, list)
    assert isinstance(det.last_oriented_bbox_screen_frame, list)
    for bbox in det.last_oriented_bbox_screen + det.last_oriented_bbox_screen_frame:
        assert np.asarray(bbox).shape == (4, 2), \
            f"oriented bbox must be 4 corner points, got shape {np.asarray(bbox).shape}"
    print(f"    {len(results)} seg detection(s); "
          f"screen={len(det.last_oriented_bbox_screen)} "
          f"screen_frame={len(det.last_oriented_bbox_screen_frame)}")


def check_real_camera():
    """RealCamera color + aligned depth snapshot + exposure roundtrip."""
    _require_module("pyrealsense2",
                    "pip install pyrealsense2 on the box machine")
    from managed_system.adapters.real import RealCamera
    try:
        cam = RealCamera(width=1280, height=720, exposure_us=2500)
    except Exception as exc:
        raise SkipCheck(f"could not open RealSense camera "
                        f"({type(exc).__name__}: {exc}) - is it plugged in?")
    try:
        img = cam.get_color_image()
        assert img.shape == (720, 1280, 3) and img.dtype == np.uint8

        snap = cam.get_depth_snapshot()
        assert snap is not None, "no depth snapshot after a color grab"
        assert set(snap) == {"depth_image", "fx", "fy", "ppx", "ppy",
                             "depth_units"}, f"bad snapshot keys: {set(snap)}"
        assert snap["depth_image"].shape == (720, 1280)
        assert snap["depth_units"] > 0

        assert cam.get_exposure() == 2500
        cam.set_exposure(400)
        assert cam.get_exposure() == 400
        cam.set_exposure(2500)
        print(f"    color {img.shape}, depth_units={snap['depth_units']}, "
              f"exposure roundtrip OK")
    finally:
        cam.pipeline.stop()


def check_real_rtde():
    """RealRTDE returns a well-formed TCP pose from the actual robot."""
    try:
        socket.create_connection((ROBOT_IP, RTDE_PORT), timeout=2).close()
    except OSError as exc:
        raise SkipCheck(f"no robot reachable at {ROBOT_IP}:{RTDE_PORT} ({exc}) "
                        "- set ROBOT_IP or connect the UR robot")

    from managed_system.adapters.real import RealRTDE
    rtde = RealRTDE(robot_ip=ROBOT_IP)
    pose = rtde.get_tcp_pose()
    assert set(pose) == {"x", "y", "z", "rx", "ry", "rz"}, f"bad pose: {pose}"
    assert all(isinstance(v, float) for v in pose.values())
    print(f"    TCP pose from {ROBOT_IP}: "
          f"[{pose['x']:.3f}, {pose['y']:.3f}, {pose['z']:.3f}]")


if __name__ == "__main__":
    run_checks(
        [
            check_detection_class_order,
            check_detector_static_config,
            check_model_weights_on_disk,
            check_original_modules_importable,
            check_calibration_data,
            check_real_detector_class_names,
            check_real_detector_contract,
            check_real_detector_seg_model,
            check_real_camera,
            check_real_rtde,
        ],
        "STEP 5: REAL adapter (field deployment) checks",
    )
