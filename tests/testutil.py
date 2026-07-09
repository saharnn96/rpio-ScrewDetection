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
import time
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

    from managed_system.adapters.lightbox import LightboxCamera
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


# ---------------------------------------------------------------------------
# Real rpclpy/redis bus helpers. The MAPLE-K nodes dispatch events on redis
# pub/sub listener threads, so tests must (a) start from a clean knowledge
# store, (b) wait for events instead of assuming synchronous dispatch, and
# (c) shut old node wirings down so their subscribers stop firing.
# 127.0.0.1, never "localhost": the IPv6 fallback stalls ~21s per connection.
# ---------------------------------------------------------------------------
REDIS_HOST, REDIS_PORT = "127.0.0.1", 6379


def get_redis():
    """Connected redis client, or raise SkipCheck when no server is running."""
    import redis
    client = redis.Redis(host=REDIS_HOST, port=REDIS_PORT,
                         socket_connect_timeout=1, decode_responses=True)
    try:
        client.ping()
    except Exception as exc:
        raise SkipCheck(f"redis not reachable on {REDIS_HOST}:{REDIS_PORT} "
                        f"({type(exc).__name__}) - start redis and re-run")
    return client


def flush_redis():
    """Wipe the knowledge store/logs so a wiring starts from clean state."""
    get_redis().flushdb()


def wait_until(predicate, timeout=15.0, interval=0.05):
    """Poll `predicate` until truthy or timeout; returns the last result."""
    deadline = time.time() + timeout
    result = predicate()
    while not result and time.time() < deadline:
        time.sleep(interval)
        result = predicate()
    return result


class EventCounter:
    """Counts bus events per key via a redis pub/sub listener thread."""

    def __init__(self, *event_keys):
        self.events = {key: [] for key in event_keys}
        self._pubsub = get_redis().pubsub(ignore_subscribe_messages=True)
        self._pubsub.subscribe(**{
            key: (lambda msg, key=key: self.events[key].append(msg["data"]))
            for key in event_keys
        })
        self._thread = self._pubsub.run_in_thread(sleep_time=0.01, daemon=True)

    def count(self, event_key):
        return len(self.events[event_key])

    def wait_for(self, event_key, count=1, timeout=15.0):
        """True once `event_key` has fired at least `count` times."""
        return bool(wait_until(lambda: self.count(event_key) >= count,
                               timeout=timeout))

    def close(self):
        # Stop and JOIN the worker before closing the pubsub socket -
        # closing first makes the poll loop die with socket errors.
        self._thread.stop()
        self._thread.join(timeout=2)
        self._pubsub.close()


_live_nodes = []


def track_nodes(nodes):
    """Remember started rpclpy nodes so the next wiring can shut them down."""
    _live_nodes.extend(nodes)


def shutdown_nodes():
    """Stop all tracked nodes' pub/sub listeners (old wirings must not keep
    reacting to events meant for the current one)."""
    while _live_nodes:
        node = _live_nodes.pop()
        try:
            node.shutdown()
        except Exception:
            pass
