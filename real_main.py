"""Real-hardware entry point.

Same wiring as `simulation.py` but with REAL adapters and an XMLRPC bridge in
place of the sim's SensorPublisher.

Layers wired up:
    Layer 3   maple_k          <-- the five Nodes (unchanged)
    Layer 2   detection_core   <-- ScrewDetectionCore   (unchanged)
    Layer 1   real_adapters    <-- RealCamera / RealRTDE / RealDetector

To run in the field:
    pip install -r requirements.txt   # + uncomment the real-hardware extras
    python real_main.py

The bridge maps the synchronous UR pendant contract (identical method names /
signatures to the original `python/screwSegmentation.py` server on port 50000)
onto the async MAPLE-K event bus:

    take_new_image_and_detect  -> publish sensor_data_received, block until
                                  Analysis publishes detection_completed
    get_detected_object_coords -> read Detections/FrameRef knowledge, transform
                                  bbox centers to base-frame coords via the
                                  helpers ported onto ScrewDetectionCore
"""

import json
import logging
import logging.handlers
import os
import threading
import traceback
import uuid
import xmlrpc.client
from xmlrpc.server import SimpleXMLRPCServer

import detection_core as dc
from detection_core import ScrewDetectionCore, DetectionClasses
from maple_k import Node, build_nodes, run_dashboard, USING_REAL_RPCLPY
from messages import Detections, FrameRef

_HERE = os.path.dirname(os.path.abspath(__file__))

# CalibData folder holding FinalTransforms/T_cam2gripper_Method_1.npz - same
# layout the original code used. Override via main(calib_folder=...).
DEFAULT_CALIB_FOLDER = os.path.join(
    _HERE, "screw_detection", "python", "camera_robot_calibration",
    "screwdriver_tcp", "CalibData",
)


# ---------------------------------------------------------------------------
# XMLRPC error logging (ported from the original python/screwSegmentation.py).
#
# SimpleXMLRPCServer discards Python tracebacks: the pendant only sees
# "<ExceptionType>:<message>". LoggingXMLRPCServer logs the full traceback
# (with a short correlation id) to xmlrpc_errors.log and puts the id + the
# last frames into the Fault string, so an operator can grep the log for
# ``[rpc:ab12cd34]`` shown on the pendant.
# ---------------------------------------------------------------------------
_RPC_LOG_FILENAME = os.path.join(_HERE, "xmlrpc_errors.log")
rpc_logger = logging.getLogger("real_main.xmlrpc")


def _configure_rpc_logger():
    if rpc_logger.handlers:
        return  # already configured
    rpc_logger.setLevel(logging.DEBUG)
    rpc_logger.propagate = False
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] [%(threadName)s] %(message)s")
    try:
        file_handler = logging.handlers.RotatingFileHandler(
            _RPC_LOG_FILENAME, maxBytes=5 * 1024 * 1024, backupCount=5
        )
        file_handler.setFormatter(fmt)
        rpc_logger.addHandler(file_handler)
    except OSError as exc:
        print(f"Warning: could not open {_RPC_LOG_FILENAME}: {exc}")
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(fmt)
    rpc_logger.addHandler(stream_handler)


def _short_repr(value, max_len=200):
    """Return a truncated repr() of ``value`` safe for logging."""
    try:
        text = repr(value)
    except Exception as exc:  # repr itself can fail (e.g. on torch tensors)
        return f"<unreprable {type(value).__name__}: {exc}>"
    if len(text) > max_len:
        text = text[:max_len] + f"...<+{len(text) - max_len} chars>"
    return text


class LoggingXMLRPCServer(SimpleXMLRPCServer):
    """SimpleXMLRPCServer that logs full tracebacks and forwards a useful
    Fault to the client (the UR controller)."""

    def _dispatch(self, method, params):
        try:
            return SimpleXMLRPCServer._dispatch(self, method, params)
        except xmlrpc.client.Fault:
            raise  # already an intentional fault - just propagate
        except Exception as exc:
            corr_id = uuid.uuid4().hex[:8]
            tb = traceback.format_exc()
            params_repr = ", ".join(_short_repr(p) for p in params)
            rpc_logger.error(
                "[rpc:%s] Exception in RPC method '%s' called with (%s):\n%s",
                corr_id, method, params_repr, tb,
            )
            frames = traceback.extract_tb(exc.__traceback__)
            tail = frames[-3:] if len(frames) > 3 else frames
            tb_short = " | ".join(
                f"{os.path.basename(f.filename)}:{f.lineno} in {f.name}"
                for f in tail
            )
            fault_msg = (
                f"[rpc:{corr_id}] {type(exc).__name__}: {exc} "
                f"(method='{method}'; trace: {tb_short}; "
                f"see {os.path.basename(_RPC_LOG_FILENAME)} for full traceback)"
            )
            raise xmlrpc.client.Fault(1, fault_msg)


# ---------------------------------------------------------------------------
# XMLRPC bridge - turns the pendant's blocking RPC call into a
# sensor_data_received event, then waits for the corresponding
# detection_completed event before returning.
# ---------------------------------------------------------------------------

# UR pendant object type -> detection class label. The screen types come from
# the seg model, whose mask/oriented-bbox path is not ported yet.
_OBJ_TYPE_TO_LABEL = {
    "holder": DetectionClasses.HOLDER.value,
    "screw": DetectionClasses.SCREW.value,
    "noscrew": DetectionClasses.NOSCREW.value,
}


class RobotBridge(Node):
    """Bridges the synchronous UR XMLRPC contract to the async event bus."""

    CYCLE_TIMEOUT_S = 120.0  # UQ over T=10 forward passes can be slow on CPU

    def __init__(self, config=None, verbose=True):
        super().__init__(config=config, verbose=verbose)
        self._name = "RobotBridge"
        self._done = threading.Event()

    def register_callbacks(self):
        # Analysis publishes this on EVERY cycle right after the Detections
        # knowledge is written (plan_executed only fires on a model swap).
        self.register_event_callback(
            event_key="detection_completed",
            callback=lambda msg: self._done.set(),
        )

    # --- pendant contract (same names/signatures as the original) ----------
    def take_new_image_and_detect(self, tcp_pose, detection_type_string,
                                  screw_fixture=False, save_detections=True,
                                  region_of_interest_type_screw=True):
        """Called by the UR pendant. Blocks until Analysis has written this
        cycle's detections to the knowledge store."""
        self._done.clear()
        # Clear last cycle's detections so a failed cycle can't serve stale coords.
        self.write_knowledge(Detections())
        payload = json.dumps({
            "detection_type": detection_type_string,
            "tcp_pose": tcp_pose,
        })
        self.publish_event(event_key="sensor_data_received", message=payload)
        if not self._done.wait(timeout=self.CYCLE_TIMEOUT_S):
            raise TimeoutError(
                f"MAPLE-K cycle did not complete in {self.CYCLE_TIMEOUT_S:.0f}s"
            )
        return None  # coords fetched via get_detected_object_coords

    def get_detected_object_coords(self, obj_type, camera_view="close"):
        """Called by the UR pendant to fetch the last-cycle detection coords.

        Returns a flat [x,y,z, x,y,z, ...] list in the robot base frame,
        sorted the same way the original get_detected_object_coords_list did.
        """
        if obj_type not in _OBJ_TYPE_TO_LABEL:
            if obj_type in ("screen", "screen_frame", "screen_center",
                            "screen_frame_center"):
                raise NotImplementedError(
                    f"obj_type={obj_type!r} needs the segmentation model, whose "
                    "mask/oriented-bbox path is not ported onto the core yet."
                )
            raise ValueError(
                f"Invalid obj_type {obj_type!r}. "
                "Choose 'holder', 'screw' or 'noscrew'."
            )

        dets = self.read_knowledge(Detections)
        frame_ref = self.read_knowledge(FrameRef)
        if dets is None or frame_ref is None or frame_ref.tcp_pose is None:
            print("No detection cycle has completed yet.")
            return []

        pose = frame_ref.tcp_pose
        if isinstance(pose, dict):
            pose = [pose["x"], pose["y"], pose["z"],
                    pose["rx"], pose["ry"], pose["rz"]]

        label = _OBJ_TYPE_TO_LABEL[obj_type]
        centers = [
            [(d["box"][0] + d["box"][2]) / 2, (d["box"][1] + d["box"][3]) / 2]
            for d in dets.items
            if d["label"] == label and d["box"] is not None
        ]
        if not centers:
            print(f"No {obj_type} detected in the last cycle.")
            return []

        return dc.CORE.get_coordinates_list(centers, pose, camera_view=camera_view)

    # --- auxiliary pendant RPCs (same surface as the original server) ------
    def get_trans_T_cam2gripper(self):
        return dc.CORE.get_translation_T_cam2gripper()

    def get_exposure_time(self):
        return dc.CORE.camera.get_exposure()

    def set_exposure_time(self, exposure_time=400):
        dc.CORE.camera.set_exposure(exposure_time)
        return None

    # The original toggled a live OpenCV display / drove on-screen timers.
    # There is no display loop in the MAPLE-K deployment (the dashboard covers
    # observability), so these are accepted-and-logged no-ops to keep existing
    # URScript programs working unchanged.
    def enable_show_detections(self):
        print("enable_show_detections: no live display in MAPLE-K mode (no-op)")
        return None

    def disable_show_detections(self):
        print("disable_show_detections: no live display in MAPLE-K mode (no-op)")
        return None

    def start_timer(self, list_of_processes=None):
        print(f"start_timer({list_of_processes}): timers not used in MAPLE-K mode (no-op)")
        return None

    def switch_process(self, new_process=None):
        print(f"switch_process({new_process}): timers not used in MAPLE-K mode (no-op)")
        return None

    def stop_timer(self):
        print("stop_timer: timers not used in MAPLE-K mode (no-op)")
        return None


def build_server(bridge, xmlrpc_port=50000, host=""):
    """Pendant-facing XMLRPC server with the exact method surface the original
    python/screwSegmentation.py `run()` registered (port 50000)."""
    server = LoggingXMLRPCServer((host, xmlrpc_port), allow_none=True,
                                 logRequests=False)
    server.RequestHandlerClass.protocol_version = "HTTP/1.1"
    server.register_function(bridge.get_trans_T_cam2gripper, "get_trans_T_cam2gripper")

    server.register_function(bridge.take_new_image_and_detect, "take_new_image_and_detect")
    server.register_function(bridge.get_detected_object_coords, "get_detected_object_coords")

    server.register_function(bridge.get_exposure_time, "get_exposure_time")
    server.register_function(bridge.set_exposure_time, "set_exposure_time")

    server.register_function(bridge.enable_show_detections, "enable_show_detections")
    server.register_function(bridge.disable_show_detections, "disable_show_detections")

    server.register_function(bridge.start_timer, "start_timer")
    server.register_function(bridge.switch_process, "switch_process")
    server.register_function(bridge.stop_timer, "stop_timer")
    return server


def main(robot_ip="192.168.1.100", xmlrpc_port=50000,
         calib_folder=DEFAULT_CALIB_FOLDER):
    logging.getLogger().setLevel(logging.INFO)
    _configure_rpc_logger()
    print("=" * 78)
    print(f"Screw-detection MAPLE-K (real hardware)  "
          f"(rpclpy={'REAL' if USING_REAL_RPCLPY else 'local shim'})")
    print("=" * 78)

    # --- Layer 1: real adapters -------------------------------------------
    from real_adapters import RealCamera, RealRTDE, RealDetector
    camera = RealCamera(width=1280, height=720, exposure_us=2500)
    rtde = RealRTDE(robot_ip=robot_ip)
    detector = RealDetector(model_id=1)

    # --- Layer 2: core -----------------------------------------------------
    core = ScrewDetectionCore(
        camera=camera, rtde=rtde, detector=detector,
        out_dir="./_prod_out",
        entropy_window_size=10, entropy_threshold=0.5,
        candidate_model_id=2, max_replans=3,
        calib_folder=calib_folder,
    )
    dc.set_core(core)

    # --- Config ------------------------------------------------------------
    config = {}
    try:
        import yaml
        cfg_path = os.path.join(_HERE, "config.yaml")
        if os.path.exists(cfg_path):
            with open(cfg_path) as f:
                config = yaml.safe_load(f) or {}
    except Exception as exc:
        print(f"(config.yaml not loaded: {exc})")

    # --- Layer 3: MAPLE-K nodes + bridge ----------------------------------
    build_nodes(config)
    bridge = RobotBridge(config.get("Monitor_Config") if config else None)
    bridge.register_callbacks()
    bridge.start()

    run_dashboard(host="127.0.0.1", port=8050, debug=False, start_trust=True)

    # --- Serve the pendant -------------------------------------------------
    server = build_server(bridge, xmlrpc_port)
    print(f"Listening on port {xmlrpc_port} ...")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("Server stopped.")


if __name__ == "__main__":
    main()
