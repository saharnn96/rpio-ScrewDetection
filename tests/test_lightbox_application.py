"""STEP 2 - Application-layer (Layer 2) checks on the light box.

Exercises ScrewDetectionCore with the light-box adapters underneath - the
same primitives the MAPLE-K nodes will call in step 3:

    capture()              MONITOR primitive (real camera -> jpg on disk)
    detect()               ANALYZE/LEGITIMATE primitive (image from disk)
    append_rolling_image / list_rolling_images (entropy window upkeep)
    is_adaptive / resolve_model_id             (model routing policy)
    log_knowledge()        KNOWLEDGE primitive (CSV log)

Run on the box:   python tests/test_lightbox_application.py
Everything is written to a throwaway temp directory.
"""

import os
import tempfile

import numpy as np

from testutil import SkipCheck, run_checks, get_camera, close_camera

import detection_core
from detection_core import ScrewDetectionCore, DetectionClasses
from lightbox_adapters import LightboxRTDE, build_lightbox_detector

OUT_DIR = tempfile.mkdtemp(prefix="lightbox_app_test_")
WINDOW = 4

_detector = None
_last_capture = None  # FrameRef dict from check_core_capture, reused by detect


def _get_detector():
    global _detector
    if _detector is None:
        _detector = build_lightbox_detector(model_id=1, T=3)
    return _detector


def _build_core(camera=None):
    return ScrewDetectionCore(
        camera=camera, rtde=LightboxRTDE(), detector=_get_detector(),
        out_dir=OUT_DIR, entropy_window_size=WINDOW,
        entropy_threshold=0.5, candidate_model_id=2, max_replans=3,
    )


def _synthetic_jpg(path, level=128):
    import cv2
    cv2.imwrite(path, np.full((720, 1280, 3), level, dtype=np.uint8))
    return path


# ---------------------------------------------------------------------------
def check_core_capture():
    """MONITOR primitive: real frame captured, saved to disk, depth snapshotted."""
    global _last_capture
    core = _build_core(camera=get_camera())
    ref = core.capture("pc_screen")

    assert set(ref) == {"frame_path", "timestamp", "brightness"}, f"bad keys: {set(ref)}"
    assert os.path.exists(ref["frame_path"]), f"frame not on disk: {ref['frame_path']}"
    assert ref["frame_path"].startswith(core.capture_dir), "frame saved outside capture_dir"
    assert isinstance(ref["brightness"], float) and ref["brightness"] > 2.0, \
        f"suspicious brightness {ref['brightness']} - light box closed?"
    assert core.last_depth is not None, \
        "capture() did not snapshot depth (needed for coordinate transforms later)"
    _last_capture = ref
    print(f"    captured {os.path.basename(ref['frame_path'])} "
          f"(brightness={ref['brightness']:.1f})")


def check_core_detect():
    """ANALYZE primitive: detect on the frame capture() wrote to disk."""
    core = _build_core(camera=None)  # detect() reads from disk, no camera call
    if _last_capture is not None:
        frame_path = _last_capture["frame_path"]
    else:
        frame_path = _synthetic_jpg(os.path.join(OUT_DIR, "synthetic.jpg"))
        print("    no camera - detecting on a synthetic frame")

    detections, screw_entropy = core.detect(frame_path, core.detector.model_id)
    assert isinstance(detections, list)
    assert screw_entropy is None or isinstance(screw_entropy, float)
    print(f"    {len(detections)} detection(s), screw_entropy={screw_entropy}")


def check_rolling_window():
    """Entropy window: only the newest `entropy_window_size` frames are kept."""
    core = _build_core(camera=None)
    for i in range(WINDOW + 3):
        core.append_rolling_image(
            _synthetic_jpg(os.path.join(OUT_DIR, f"roll_{i:02d}.jpg")))

    rolling = core.list_rolling_images()
    assert len(rolling) == WINDOW, f"expected {WINDOW} rolling images, got {len(rolling)}"
    names = [os.path.basename(p) for p in rolling]
    assert names == [f"roll_{i:02d}.jpg" for i in range(3, WINDOW + 3)], \
        f"oldest frames were not evicted first: {names}"
    assert all(os.path.exists(p) for p in rolling)


def check_model_routing():
    """Task -> model routing: adaptive closeup task vs fixed per-task models."""
    core = _build_core(camera=None)
    det = core.detector

    # The MAPLE-K-owned active id must win for adaptive detection types.
    assert core.is_adaptive("pc_screen"), "'pc_screen' must be the adaptive task"
    assert core.resolve_model_id("pc_screen", active_model_id=2) == 2

    if hasattr(det, "DETECTION_TYPE_TO_MODEL"):  # real detector
        assert not core.is_adaptive("topview")
        assert core.resolve_model_id("topview", active_model_id=2) == \
            det.DETECTION_TYPE_TO_MODEL["topview"], "topview must use its fixed model"
        print("    real detector routing verified (adaptive closeup, fixed topview)")
    else:
        assert core.is_adaptive("topview"), \
            "sim detector declares no routing tables - every task is adaptive"
        print("    sim detector fallback: every detection type is adaptive")


def check_log_knowledge():
    """KNOWLEDGE primitive: cycle summary appended to the CSV log."""
    import csv
    core = _build_core(camera=None)
    detections = [
        {"label": DetectionClasses.HOLDER.value, "entropy": 0.05},
        {"label": DetectionClasses.SCREW.value, "entropy": 0.12},
    ]
    core.log_knowledge("20260707_000000", {"brightness": 123.4}, detections)

    assert os.path.exists(core.knowledge_log_path), "knowledge_log.csv not created"
    with open(core.knowledge_log_path, newline="") as f:
        rows = list(csv.reader(f))
    assert rows[0] == ["timestamp", "brightness", "active_model",
                       "num_detections", "screw_entropy"], f"bad header: {rows[0]}"
    last = rows[-1]
    assert last[0] == "20260707_000000" and last[3] == "2", f"bad row: {last}"
    assert last[4] == "0.1200", f"screw entropy not logged: {last}"


if __name__ == "__main__":
    try:
        run_checks(
            [
                check_core_capture,
                check_core_detect,
                check_rolling_window,
                check_model_routing,
                check_log_knowledge,
            ],
            f"STEP 2: light-box APPLICATION checks (out dir: {OUT_DIR})",
        )
    finally:
        close_camera()
