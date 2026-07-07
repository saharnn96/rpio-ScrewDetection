"""Shared helpers for the light-box test scripts.

The tests are plain Python scripts (no pytest dependency) so they run on the
educational box with nothing but the repo's requirements installed:

    python tests/test_lightbox_adapters.py     # step 1: adapter functions
    python tests/test_lightbox_application.py  # step 2: ScrewDetectionCore
    python tests/test_lightbox_maple_k.py      # step 3: full MAPLE-K loop

Checks that need hardware that isn't present are reported as [SKIP], not
[FAIL], so the scripts are still useful on a dev machine without the camera.
"""

import os
import sys
import traceback

# Make the repo root importable no matter where the script is launched from.
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)


class SkipCheck(Exception):
    """Raise inside a check to report [SKIP] with a reason."""


def run_checks(checks, title):
    """Run check functions in order; print PASS/SKIP/FAIL; exit 1 on failure."""
    print("=" * 70)
    print(title)
    print("=" * 70)
    passed, skipped, failed = [], [], []
    for fn in checks:
        name = fn.__name__
        print(f"\n--- {name} ---")
        try:
            fn()
        except SkipCheck as exc:
            skipped.append(name)
            print(f"[SKIP] {name}: {exc}")
        except Exception as exc:
            failed.append(name)
            traceback.print_exc()
            print(f"[FAIL] {name}: {exc}")
        else:
            passed.append(name)
            print(f"[PASS] {name}")

    print("\n" + "=" * 70)
    print(f"Result: {len(passed)} passed, {len(skipped)} skipped, "
          f"{len(failed)} failed")
    if skipped:
        print("Skipped:", ", ".join(skipped))
    if failed:
        print("FAILED :", ", ".join(failed))
    print("=" * 70)
    sys.exit(1 if failed else 0)


# ---------------------------------------------------------------------------
# Shared camera handling. The RealSense pipeline must only be opened once per
# process, so every check that needs the camera goes through get_camera().
# ---------------------------------------------------------------------------
_camera = None
_camera_error = None


def get_camera():
    """Open (once) and return the real LightboxCamera, or raise SkipCheck."""
    global _camera, _camera_error
    if _camera is not None:
        return _camera
    if _camera_error is not None:
        raise SkipCheck(_camera_error)

    try:
        import pyrealsense2  # noqa: F401
    except ImportError:
        _camera_error = ("pyrealsense2 not installed - "
                         "run `pip install pyrealsense2` on the box machine")
        raise SkipCheck(_camera_error)

    from lightbox_adapters import LightboxCamera
    try:
        _camera = LightboxCamera(width=1280, height=720, exposure_us=2500)
    except Exception as exc:
        _camera_error = (f"could not open RealSense camera "
                         f"({type(exc).__name__}: {exc}) - is it plugged in?")
        raise SkipCheck(_camera_error)
    return _camera


def close_camera():
    global _camera
    if _camera is not None:
        _camera.close()
        _camera = None


def reset_bus():
    """Fresh in-process event bus + knowledge store (needed between MAPLE-K
    wirings, since nodes register callbacks on a module-level singleton)."""
    import maple_k
    if maple_k.USING_REAL_RPCLPY:
        raise SkipCheck("real rpclpy/redis bus is active; these loop tests "
                        "need the in-process shim - stop redis and re-run")
    import local_bus
    local_bus.BUS = local_bus._Bus()
