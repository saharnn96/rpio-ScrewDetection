"""STEP 1 - Light-box adapter checks.

Verifies every function on each Layer-1 adapter in isolation, before anything
above it (core, MAPLE-K) is involved:

    LightboxCamera : get_color_image / get_depth_snapshot / get_exposure /
                     set_exposure
    LightboxRTDE   : get_tcp_pose / set_tcp_pose
    detector       : detect() return contract

Run on the box:   python tests/test_lightbox_adapters.py
A snapshot of what the camera sees is saved next to this script as
`lightbox_snapshot.jpg` - open it to confirm the camera is aimed correctly.
"""

import os

import numpy as np

from testutil import SkipCheck, run_checks, get_camera, close_camera, REPO_ROOT

from detection_core import DetectionResult
from lightbox_adapters import LightboxRTDE, build_lightbox_detector

SNAPSHOT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "lightbox_snapshot.jpg")


# ---------------------------------------------------------------------------
# LightboxRTDE (mock) - no hardware needed, must always pass.
# ---------------------------------------------------------------------------
def check_rtde_get_tcp_pose():
    rtde = LightboxRTDE()
    pose = rtde.get_tcp_pose()
    assert isinstance(pose, dict), f"pose must be a dict, got {type(pose)}"
    assert set(pose) == {"x", "y", "z", "rx", "ry", "rz"}, f"bad keys: {set(pose)}"
    for k, v in pose.items():
        assert isinstance(v, float), f"pose[{k!r}] must be float, got {type(v)}"
    # Must return a copy - mutating the result must not corrupt the adapter.
    pose["x"] = 999.0
    assert rtde.get_tcp_pose()["x"] != 999.0, "get_tcp_pose leaked internal state"


def check_rtde_set_tcp_pose():
    rtde = LightboxRTDE()
    rtde.set_tcp_pose(x=0.1, z=0.25)
    pose = rtde.get_tcp_pose()
    assert pose["x"] == 0.1 and pose["z"] == 0.25, f"set_tcp_pose not applied: {pose}"
    try:
        rtde.set_tcp_pose(bogus=1.0)
    except ValueError:
        pass
    else:
        raise AssertionError("set_tcp_pose accepted an unknown component")


# ---------------------------------------------------------------------------
# LightboxCamera (real RealSense) - skipped cleanly if no camera present.
# ---------------------------------------------------------------------------
def check_camera_color_image():
    import cv2
    cam = get_camera()
    frame = cam.get_color_image()
    assert isinstance(frame, np.ndarray), f"expected ndarray, got {type(frame)}"
    assert frame.dtype == np.uint8, f"expected uint8, got {frame.dtype}"
    assert frame.shape == (720, 1280, 3), f"expected (720,1280,3), got {frame.shape}"
    brightness = float(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).mean())
    print(f"    frame ok, mean brightness = {brightness:.1f}")
    assert brightness > 2.0, ("frame is essentially black - lens cap on / "
                              "light box closed / exposure too low?")
    cv2.imwrite(SNAPSHOT_PATH, frame)
    print(f"    snapshot saved to {SNAPSHOT_PATH}")


def check_camera_depth_snapshot():
    cam = get_camera()
    cam.get_color_image()  # depth snapshot is aligned to the last color frame
    depth = cam.get_depth_snapshot()
    assert depth is not None, "get_depth_snapshot returned None after a color grab"
    expected = {"depth_image", "fx", "fy", "ppx", "ppy", "depth_units"}
    assert set(depth) == expected, f"bad keys: {set(depth)}"
    img = depth["depth_image"]
    assert isinstance(img, np.ndarray) and img.shape == (720, 1280), \
        f"depth image should be 720x1280, got {getattr(img, 'shape', None)}"
    assert depth["depth_units"] > 0, "depth_units must be positive"
    coverage = float((img > 0).mean())
    print(f"    depth ok, units={depth['depth_units']:.6f} m/unit, "
          f"valid-pixel coverage={coverage:.0%}")


def check_camera_exposure_roundtrip():
    cam = get_camera()
    original = cam.get_exposure()
    assert original == 2500, f"expected initial exposure 2500, got {original}"
    try:
        cam.set_exposure(1000)
        assert cam.get_exposure() == 1000, "set_exposure(1000) did not stick"
        cam.get_color_image()  # camera must still stream at the new exposure
    finally:
        cam.set_exposure(original)
    assert cam.get_exposure() == original


# ---------------------------------------------------------------------------
# Detector - real YOLO+UQ if installed, otherwise sim fallback (still checks
# the return contract that Layer 2 depends on).
# ---------------------------------------------------------------------------
def check_detector_contract():
    detector = build_lightbox_detector(model_id=1, T=3)
    from sim_adapters import SimulatedDetector
    if isinstance(detector, SimulatedDetector):
        print("    NOTE: using SimulatedDetector fallback (install "
              "ultralytics/torch/deepluq on the box for real detection)")

    assert hasattr(detector, "model_id"), "detector must expose model_id"

    # Prefer a real frame from the box; fall back to a synthetic one.
    try:
        image = get_camera().get_color_image()
        print("    detecting on a live light-box frame")
    except SkipCheck:
        image = np.full((720, 1280, 3), 128, dtype=np.uint8)
        print("    no camera - detecting on a synthetic grey frame")

    results, screw_entropy = detector.detect(image, detector.model_id)

    assert isinstance(results, list), f"detect must return a list, got {type(results)}"
    for r in results:
        assert isinstance(r, DetectionResult), f"bad result type {type(r)}"
        assert isinstance(r.label, int), f"label must be int, got {type(r.label)}"
        assert r.box is not None and len(r.box) == 4, f"bad box: {r.box}"
        assert 0.0 <= float(r.score) <= 1.0, f"score out of range: {r.score}"
        assert r.entropy is None or 0.0 <= float(r.entropy) <= 1.0, \
            f"entropy out of range: {r.entropy}"
    assert screw_entropy is None or isinstance(screw_entropy, float), \
        f"screw_entropy must be float|None, got {type(screw_entropy)}"
    print(f"    {len(results)} detection(s), screw_entropy={screw_entropy}")
    if not results:
        print("    NOTE: 0 detections is contract-valid; put a screw/holder in "
              "the box to see real detections")


if __name__ == "__main__":
    try:
        run_checks(
            [
                check_rtde_get_tcp_pose,
                check_rtde_set_tcp_pose,
                check_camera_color_image,
                check_camera_depth_snapshot,
                check_camera_exposure_roundtrip,
                check_detector_contract,
            ],
            "STEP 1: light-box ADAPTER checks (camera real, robot mocked)",
        )
    finally:
        close_camera()
