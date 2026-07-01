"""In-process fallback for the rpclpy `Node` API.

`screwSegmentation_rpio.py` is written against the real rpclpy API
(`from rpclpy.node import Node`, `publish_event`, `read_knowledge`, ...).
rpclpy is not installed in this simulation environment and it requires a
running redis server, so this module provides a drop-in, dependency-free
replacement that runs everything in a single process.

Differences from real rpclpy (intentional, for offline simulation):
  * The event bus is SYNCHRONOUS: `publish_event(k)` invokes every registered
    callback for `k` inline, on the caller's stack. So one `SensorData` emit
    drives the whole Monitor->Analysis->Plan->Legitimate->Execute chain to
    completion before returning. With real redis the same chain runs
    asynchronously across processes; the per-node logic is identical.
  * The knowledge store is a plain in-memory dict keyed by class name. With
    real redis it is serialized JSON in a shared db.

To run distributed for real: `pip install rpclpy redis`, start redis, and the
`try: from rpclpy...` import at the top of screwSegmentation_rpio.py will pick
up the real library instead of this shim.
"""

import functools
import logging
import time


class _Bus:
    """Singleton event bus + knowledge store shared by all nodes in-process."""

    def __init__(self):
        self.subscribers = {}   # event_key -> list[callback]
        self.knowledge = {}     # class __name__ -> object instance

    def subscribe(self, event_key, callback):
        self.subscribers.setdefault(event_key, []).append(callback)

    def publish(self, event_key, message=None):
        for callback in list(self.subscribers.get(event_key, [])):
            callback(message)

    def write(self, obj):
        self.knowledge[type(obj).__name__] = obj

    def read(self, cls):
        return self.knowledge.get(cls.__name__)


# Shared across every Node instance (mirrors the shared redis db).
BUS = _Bus()


def timeit_callback(fn):
    """Stand-in for rpclpy.utils.timeit_callback - times and logs the call."""
    @functools.wraps(fn)
    def wrapper(self, *args, **kwargs):
        start = time.time()
        result = fn(self, *args, **kwargs)
        elapsed = (time.time() - start) * 1000.0
        logger = getattr(self, "logger", logging.getLogger(__name__))
        logger.debug("%s took %.1f ms", fn.__name__, elapsed)
        return result
    return wrapper


def run_dashboard(**kwargs):
    """Stand-in for rpclpy.DashboardApp.run_dashboard (no-op offline)."""
    print("[local_bus] rpclpy dashboard unavailable offline; skipping dashboard.")


class Node:
    """Minimal re-implementation of rpclpy.node.Node for offline simulation."""

    def __init__(self, config=None, verbose=True):
        self.config = config or {}
        self.verbose = verbose
        self._name = self.__class__.__name__
        self.logger = logging.getLogger(self._name)
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(logging.Formatter(
                "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
            ))
            self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO)

    # --- knowledge store -------------------------------------------------
    def write_knowledge(self, obj):
        BUS.write(obj)

    def read_knowledge(self, cls):
        return BUS.read(cls)

    # --- event bus -------------------------------------------------------
    def publish_event(self, event_key, message=None):
        if self.verbose:
            self.logger.info("--> publish_event('%s')", event_key)
        BUS.publish(event_key, message)

    def register_event_callback(self, event_key, callback):
        BUS.subscribe(event_key, callback)

    def start(self):
        # Real rpclpy spins up redis subscriber threads here; nothing to do offline.
        pass
