"""Diagnose why a candidate model scores badly in Legitimate.

Legitimate averages screw/noscrew entropy over the rolling frames, counting a
frame with NO surviving detection as 1.0. A candidate average near 1.0 (e.g.
the `cand 0.950` rejects in the log) therefore means "the model detected
nothing on almost every rolling frame" - this script shows WHY, per frame:

  raw        everything the YOLO model sees at conf 0.25, before any filtering
  holders    the holder pre-pass (conf >= 0.75) that gates closeup detections
  survived   what detect() actually returns after the entropy (<= 0.5) and
             holder-containment filters

Run on the box (needs ultralytics/torch/deepluq and the box weights):

    python tests/debug_candidate_frames.py [model_id] [rolling_dir]

Defaults: model_id=2 (the candidate), rolling_dir from lightbox_main's
out_dir (./_lightbox_out/entropy_rolling_average_images).
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from managed_system.adapters.lightbox import LightboxDetector

DEFAULT_ROLLING = os.path.join(".", "_lightbox_out",
                               "entropy_rolling_average_images")


def main(model_id=2, rolling_dir=DEFAULT_ROLLING):
    import cv2

    frames = sorted(os.listdir(rolling_dir)) if os.path.isdir(rolling_dir) else []
    if not frames:
        print(f"No rolling frames found in {os.path.abspath(rolling_dir)} - "
              "run the loop first (or pass the rolling dir as 2nd argument).")
        return

    try:
        detector = LightboxDetector(model_id=model_id)
    except ImportError as exc:
        print(f"Cannot load the real detector here ({exc}).\n"
              "This diagnostic needs the ML stack - run it on the box:\n"
              "    pip install ultralytics torch torchmetrics deepluq")
        return
    model = detector._models[model_id]
    names = model.names  # class id -> name

    print(f"Scoring {len(frames)} rolling frame(s) with model_id={model_id} "
          f"({detector.MODEL_PATHS.get(model_id, '?')})\n")

    misses = 0
    for fname in frames:
        path = os.path.join(rolling_dir, fname)
        image = cv2.imread(path)
        print(f"=== {fname} ===")

        # 1. Raw view: everything the model sees at a permissive confidence.
        raw = model.predict(image, conf=0.25, verbose=False)[0]
        if len(raw.boxes) == 0:
            print("  raw:      NOTHING at conf 0.25 - model is blind on this frame")
        for box in raw.boxes:
            cls = int(box.cls)
            print(f"  raw:      {names.get(cls, cls):<12} conf={float(box.conf[0]):.2f}")

        # 2. Holder pre-pass exactly as detect() runs it for closeup models.
        if model_id in detector.CLOSEUP_MODEL_IDS:
            holders = detector._detect_holders(image, model_id)
            if holders:
                for h in holders:
                    print(f"  holders:  holder conf={h.score:.2f}  box={h.box.astype(int).tolist()}")
            else:
                print("  holders:  NONE at conf 0.75 -> ALL screw/noscrew "
                      "detections on this frame will be discarded")

        # 3. What actually survives the full detect() pipeline.
        results, screw_entropy = detector.detect(image, model_id)
        for r in results:
            print(f"  survived: label={r.label} score={r.score:.2f} "
                  f"entropy={'n/a' if r.entropy is None else f'{r.entropy:.3f}'}")
        if screw_entropy is None:
            misses += 1
            print("  -> screw_entropy=None  (counts as 1.0 in Legitimate)")
        else:
            print(f"  -> screw_entropy={screw_entropy:.3f}")
        print()

    n = len(frames)
    print(f"Summary: {misses}/{n} frames had no surviving screw/noscrew "
          f"detection -> Legitimate penalty floor = {misses / n:.3f}")


if __name__ == "__main__":
    mid = int(sys.argv[1]) if len(sys.argv) > 1 else 2
    rdir = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_ROLLING
    main(mid, rdir)
