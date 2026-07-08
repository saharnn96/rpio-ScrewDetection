"""Layer 1 (Hardware layer) - LIGHTBOX adapters for the educational light-box setup.

Standalone module: it does NOT import anything from `real.py`, so the box code
can evolve independently of the field-deployment adapters. It only depends on
the shared core contract (`managed_system.core`) and, lazily, on the hardware/
ML libraries themselves.

The light box has a REAL RealSense camera but NO robot yet, so this module
mixes real adapters with one mock:

  * LightboxCamera  - real pyrealsense2 pipeline (aligned color+depth,
                      exposure control) plus a `close()` helper so test
                      scripts can release the device.
  * LightboxRTDE    - mocked robot telemetry. Same interface as the real RTDE
                      adapter (`get_tcp_pose()`), returns a fixed-but-settable
                      pose. There is no outgoing "move robot" adapter in this
                      architecture - robot commands arrive FROM the UR pendant
                      via XMLRPC (see real_main.py) - so mocking telemetry is
                      all that is needed until the robot arrives.
  * LightboxDetector - YOLO + MC-dropout UQ detector loaded from the BOX
                      models: the weights vendored in this repo under
                      `screw_detection/detection_model` (the fixture-trained
                      runs the light box was set up with).
  * build_lightbox_detector() - LightboxDetector if the ML stack
                      (ultralytics/torch/deepluq) is installed, otherwise the
                      SimulatedDetector with a loud warning. On the box you
                      want the real one: `pip install ultralytics torch
                      torchmetrics deepluq pyrealsense2`.

Wire-up entry point: `lightbox_main.py`. Test scripts: `tests/`.
"""

import os

import numpy as np

from managed_system.core import (
    DetectionResult, DetectionClasses, filter_detections_within_holders,
)

# Repo root (this file lives at managed_system/adapters/lightbox.py). The box
# model weights are resolved against this so the scripts work no matter what
# CWD they are launched from.
_REPO_ROOT = os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))
_DETECTION_MODEL_DIR = os.path.join(
    _REPO_ROOT, "screw_detection", "detection_model"
)
# Box-specific weights shipped next to this module.
_LIGHTBOX_MODELS_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "lightbox_models"
)


def _box_weights(*parts):
    """Absolute path to a detect-run best.pt under the repo's detection_model."""
    return os.path.join(_DETECTION_MODEL_DIR, "scripts", "runs", "detect",
                        *parts, "weights", "best.pt")


# ---------------------------------------------------------------------------
# CameraAdapter - REAL RealSense camera (aligned color+depth).
# ---------------------------------------------------------------------------
class LightboxCamera:
    """Real RealSense camera for the light box.

    Same init/frame-grab behaviour as the field camera - the point of the box
    is to exercise the exact camera path that will run in the field - plus
    `close()` so tests can start/stop the pipeline repeatedly.
    """

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

    # --- exposure control -----------------------------------------------
    def get_exposure(self):
        return self.exposure_us

    def set_exposure(self, exposure_us):
        import time
        self.exposure_us = exposure_us
        self._sensor.set_option(self._rs.option.exposure, exposure_us)
        print(f"Exposure time set to: {exposure_us}")
        time.sleep(0.2)  # let the camera adjust

    def close(self):
        try:
            self.pipeline.stop()
        except Exception:
            pass  # already stopped / never started


# ---------------------------------------------------------------------------
# RTDEAdapter - MOCKED robot telemetry (no robot on the box yet).
# ---------------------------------------------------------------------------
class LightboxRTDE:
    """Stands in for the real RTDE adapter until the box gets a robot.

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
# DetectorAdapter - YOLO + MC-Dropout UQ on the box models.
# ---------------------------------------------------------------------------
class LightboxDetector:
    """YOLO + run_uq + uq_analysis on the BOX models (the weights vendored in
    this repo under screw_detection/detection_model). Owns one loaded model
    per model_id.

    The `detect()` return contract is identical to `SimulatedDetector.detect`:
        (list[DetectionResult], screw_entropy: float | None)
    """

    ENTROPY_THRESHOLD = 0.5  # UQ-path keep filter, matches original uq_analysis()
    PLAIN_KEEP_ENTROPY = 0.75  # plain-path keep filter (= predict conf 0.25)

    # Box model id space (absolute paths, anchored to the repo so they resolve
    # regardless of CWD). Ids 1 & 2 are the two variants the MAPLE-K loop
    # swaps between (ActiveModel.model_id carries these) - on the box these
    # the weights shipped in lightbox_models/. Ids 3-5 are the fixed per-task
    # models, selected by detection_type and never touched by the swap logic.
    MODEL_PATHS = {
        1: os.path.join(_LIGHTBOX_MODELS_DIR, "leuven.pt"),
        2: os.path.join(_LIGHTBOX_MODELS_DIR, "model2.pt"),
        3: _box_weights("real_images_with_noscrews_and_screw_fixture_extended"),
        5: _box_weights("real_images_with_noscrews_and_screw_fixture"),
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
    # outside a holder are discarded, like the original PC_CLOSEUP path -
    # but only if the model actually has a holder class (the box models
    # don't; see CLASS_NAME_MAP).
    CLOSEUP_MODEL_IDS = frozenset({1, 2})

    # Model class NAME -> pipeline DetectionClasses value. Each weights file
    # ships its own class schema (the box models use {0: screw, 1: hole};
    # the original field models {0: holder, 1: noscrew, 2: screw}), so labels
    # are translated by name. Unknown names pass through unchanged.
    CLASS_NAME_MAP = {
        "holder": DetectionClasses.HOLDER.value,
        "noscrew": DetectionClasses.NOSCREW.value,
        "no_screw": DetectionClasses.NOSCREW.value,
        "hole": DetectionClasses.NOSCREW.value,  # empty hole = missing screw
        "screw": DetectionClasses.SCREW.value,
    }

    # Bbox-detection ids that run through the run_uq / entropy path. The seg
    # model (id 4) is intentionally excluded.
    _BBOX_MODEL_IDS = frozenset(MODEL_PATHS)

    def __init__(self, model_id=1, T=10, uq_config=(0.05,), model_paths=None):
        """`model_paths` optionally overrides MODEL_PATHS entries (dict of
        model_id -> weights path); relative paths are resolved against the
        repo root."""
        from ultralytics import YOLO

        # The MC-dropout UQ path needs torchmetrics/deepluq AND the project's
        # patched ultralytics (stock ultralytics has no `Results.detection`).
        # Probe the imports here; the patched-fork check can only happen at
        # the first detect(), which drops to the plain path if it fails.
        try:
            from managed_system.uq import run_uq  # noqa: F401
            self._uq_available = True
        except ImportError as exc:
            self._uq_available = False
            print(f"NOTE: UQ stack unavailable ({exc}); using plain YOLO "
                  "detection with pseudo-entropy = 1 - confidence.")

        self.model_id = model_id
        self.T = T
        self.uq_config = list(uq_config)
        paths = dict(self.MODEL_PATHS)
        for mid, path in (model_paths or {}).items():
            paths[int(mid)] = os.path.join(_REPO_ROOT, path)
        self._models = {}
        # Dedup by path: several ids share the same weights file today, so we
        # only load each file once.
        loaded_by_path = {}
        for mid, path in paths.items():
            if path not in loaded_by_path:
                m = YOLO(path)
                m.fuse()
                loaded_by_path[path] = m
            self._models[mid] = loaded_by_path[path]

        # Per-model label translation (see CLASS_NAME_MAP) + whether the
        # model can do holder gating at all.
        self._label_maps, self._has_holder = {}, {}
        for mid, m in self._models.items():
            names = getattr(m, "names", None) or {}
            lmap = {int(cid): self.CLASS_NAME_MAP.get(str(n).lower(), int(cid))
                    for cid, n in names.items()}
            self._label_maps[mid] = lmap
            self._has_holder[mid] = (
                DetectionClasses.HOLDER.value in lmap.values())

    def _map_label(self, model_id, cls_id):
        return self._label_maps[model_id].get(int(cls_id), int(cls_id))

    def detect(self, image, model_id):
        if model_id not in self._BBOX_MODEL_IDS:
            # Seg model (screen/screen_frame): masks, no UQ, and coordinate
            # extraction via oriented bboxes - none of that is ported onto the
            # core yet. Fail loudly rather than silently mis-detecting.
            raise NotImplementedError(
                f"model_id={model_id} is a segmentation model; the mask/coord "
                "path is not ported onto ScrewDetectionCore yet."
            )

        results = None
        if self._uq_available:
            try:
                results = self._detect_uq(image, model_id)
            except AttributeError as exc:
                # Stock ultralytics: run_uq needs `Results.detection` (raw
                # logits), which only the project's patched fork provides.
                # Drop to the plain path for the rest of the run.
                self._uq_available = False
                print("=" * 70)
                print(f"WARNING: UQ path unusable "
                      f"({str(exc).splitlines()[0]})")
                print("This ultralytics build lacks the patched "
                      "Results.detection field. Falling back to plain YOLO "
                      "detection with pseudo-entropy = 1 - confidence.")
                print("=" * 70)
        if results is None:
            results = self._detect_plain(image, model_id)

        if model_id in self.CLOSEUP_MODEL_IDS and self._has_holder[model_id]:
            # Closeup task, like the original PC_CLOSEUP path: screws/noscrews
            # come from the UQ pass, holders from a plain confidence-filtered
            # pre-pass, and screw/noscrew detections whose center is outside
            # every holder are discarded. Skipped entirely for models without
            # a holder class (the current box models).
            holders = self._detect_holders(image, model_id)
            results = [r for r in results
                       if r.label != DetectionClasses.HOLDER.value] + holders
            results = filter_detections_within_holders(results)

        # Entropy over the surviving screw/noscrew detections (post-filter, so
        # a stray detection outside the holder can't feed the entropy window).
        # The original closeup scene had exactly ONE screw; the box scene has
        # many, so average over the survivors - identical to the original
        # behaviour when a single screw is present, robust when not.
        entropies = [r.entropy for r in results
                     if r.label in (DetectionClasses.SCREW.value,
                                    DetectionClasses.NOSCREW.value)
                     and r.entropy is not None]
        screw_entropy = float(np.mean(entropies)) if entropies else None

        return results, screw_entropy

    def _detect_uq(self, image, model_id):
        """MC-dropout UQ pass (T stochastic predictions + WBF clustering).
        Needs the patched ultralytics; raises AttributeError on stock."""
        from managed_system.uq import run_uq

        uq_preds = run_uq(self._models[model_id], image, T=self.T,
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
                label=self._map_label(model_id, d["label"]),
                box=np.asarray(d["box"], dtype=float),
                score=float(d["score"]),
                mask=d.get("mask"),
                entropy=float(entropy),
            ))
        return results

    def _detect_plain(self, image, model_id):
        """Single plain YOLO pass for stock ultralytics - no MC dropout, no
        logits. Uncertainty is approximated as pseudo-entropy = 1 - confidence
        so the entropy filter and the MAPLE-K window keep working on the same
        [0, ENTROPY_THRESHOLD] scale as the UQ path."""
        preds = self._models[model_id].predict(image, conf=0.25, verbose=False)

        results = []
        for box in preds[0].boxes:
            score = float(box.conf[0])
            entropy = 1.0 - score
            if entropy > self.PLAIN_KEEP_ENTROPY:
                continue
            # NOTE: deliberately looser than the UQ path's ENTROPY_THRESHOLD.
            # If uncertain detections were dropped at the anomaly threshold
            # (0.5), the entropy window could never average above it and
            # MAPLE-K adaptation would be unreachable. Keeping detections up
            # to 0.75 lets degraded confidence actually show in the window.
            results.append(DetectionResult(
                label=self._map_label(model_id, box.cls),
                box=box.xyxy[0].cpu().numpy().astype(float),
                score=score,
                mask=None,
                entropy=entropy,
            ))
        return results

    def _detect_holders(self, image, model_id):
        """Plain (non-UQ) holder pre-pass: predict at conf 0.75 and keep only
        HOLDER boxes. Port of the original's holder detection in
        take_new_image_and_detect()."""
        preds = self._models[model_id].predict(
            image, conf=0.75, verbose=False, iou=0.1
        )
        holders = []
        for box in preds[0].boxes:
            if self._map_label(model_id, box.cls) != DetectionClasses.HOLDER.value:
                continue
            holders.append(DetectionResult(
                label=DetectionClasses.HOLDER.value,
                box=box.xyxy[0].cpu().numpy().astype(float),
                score=float(box.conf[0]),
                mask=None,
                entropy=None,
            ))
        return holders


def build_lightbox_detector(model_id=1, allow_sim_fallback=True, **detector_kwargs):
    """Return the best detector available on this machine.

    Tries LightboxDetector (YOLO + MC-dropout UQ on the box models, needs
    ultralytics/torch/deepluq and the weight files under
    screw_detection/detection_model). If that fails and `allow_sim_fallback`
    is True, returns SimulatedDetector so the rest of the stack can still be
    exercised.

    `detector_kwargs` are forwarded to LightboxDetector (e.g. T=3 to speed up
    the MC-dropout passes during testing).
    """
    try:
        return LightboxDetector(model_id=model_id, **detector_kwargs)
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
