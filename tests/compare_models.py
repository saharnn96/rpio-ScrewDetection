"""Live model viewer - see detections in a camera window, switch models live.

Streams the light-box camera, runs the currently selected model on every
frame and shows the annotated video in a window. Change the lighting, flip
between models, and watch which one keeps seeing the screws.

Run on the box:

    python tests/compare_models.py

Keys (with the camera window focused):
    n / space   next model
    1..9        jump straight to model N
    q / Esc     quit

Models are auto-discovered from managed_system/adapters/lightbox_models/.
"""

import glob
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(_REPO_ROOT,
                          "managed_system", "adapters", "lightbox_models")
WINDOW = "box models - live"


def _anomaly_threshold(default=0.35):
    """entropy_threshold from managing_system/config.yaml, so the overlay
    always shows the same anomaly limit the live MAPLE-K loop uses."""
    try:
        import yaml
        with open(os.path.join(_REPO_ROOT, "managing_system", "config.yaml")) as f:
            return float(yaml.safe_load(f)["Adaptation_Config"]["entropy_threshold"])
    except Exception:
        return default


ANOMALY_THRESHOLD = _anomaly_threshold()


def main():
    import cv2
    from ultralytics import YOLO
    from managed_system.adapters.lightbox import LightboxCamera

    weights = sorted(glob.glob(os.path.join(MODELS_DIR, "*.pt")))
    if not weights:
        sys.exit(f"No .pt files found in {MODELS_DIR}")

    models = []
    for path in weights:
        name = os.path.splitext(os.path.basename(path))[0]
        print(f"Loading {name} ...")
        models.append((name, YOLO(path)))

    idx = 0
    cam = LightboxCamera()
    print("\nStreaming. Keys: n/space = next model, 1..9 = pick model, q = quit")
    try:
        while True:
            frame = cam.get_color_image()
            name, model = models[idx]
            result = model.predict(frame, conf=0.25, verbose=False)[0]
            view = result.plot()

            cv2.putText(view,
                        f"[{idx + 1}/{len(models)}] {name}   "
                        f"{len(result.boxes)} detection(s)",
                        (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                        (0, 255, 0), 2)

            # Same pseudo-entropy the MAPLE-K loop uses: 1 - confidence,
            # averaged over all detections (the window stat); worst shown too.
            # Anomaly triggers when the WINDOW AVERAGE of the mean exceeds the
            # entropy_threshold in managing_system/config.yaml.
            confs = [float(b.conf[0]) for b in result.boxes]
            if confs:
                mean_e = 1.0 - sum(confs) / len(confs)
                worst_e = 1.0 - min(confs)
                color = (0, 0, 255) if mean_e > ANOMALY_THRESHOLD else (0, 255, 0)
                text = (f"entropy mean={mean_e:.2f} worst={worst_e:.2f}  "
                        f"(anomaly if avg > {ANOMALY_THRESHOLD})")
            else:
                color = (0, 0, 255)
                text = "entropy: n/a - nothing detected"
            cv2.putText(view, text, (10, 75), cv2.FONT_HERSHEY_SIMPLEX,
                        1.0, color, 2)
            cv2.imshow(WINDOW, view)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):  # q or Esc
                break
            if key in (ord("n"), ord(" ")):
                idx = (idx + 1) % len(models)
            elif ord("1") <= key <= ord("9") and key - ord("1") < len(models):
                idx = key - ord("1")
    finally:
        cam.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
