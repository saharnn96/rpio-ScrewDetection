"""Real-hardware entry point.

Same wiring as `simulation.py` but with REAL adapters and an XMLRPC bridge in
place of the sim's SensorPublisher.

Layers wired up:
    Layer 3   maple_k          <-- the five Nodes (unchanged)
    Layer 2   screwSegmentation <-- ScrewDetectionCore   (unchanged)
    Layer 1   real_adapters    <-- RealCamera / RealRTDE / RealDetector

To run in the field:
    pip install -r requirements.txt   # + uncomment the real-hardware extras
    python real_main.py

NOTE - The XMLRPC bridge here is a SKELETON. It shows how the synchronous UR
pendant call gets mapped to a sensor_data_received event and how the caller is
unblocked when Execute publishes `plan_executed`. You will still need to
port the coordinate-transform helpers from the original
`python/screwSegmentation.py` (`calc_img_point_to_base_frame`,
`get_coordinates_list`, etc.) onto the core so `get_detected_object_coords`
has something to return.
"""

import json
import logging
import os
import threading
from xmlrpc.server import SimpleXMLRPCServer

import screwSegmentation as ss
from screwSegmentation import ScrewDetectionCore
from maple_k import Node, build_nodes, run_dashboard, USING_REAL_RPCLPY


# ---------------------------------------------------------------------------
# XMLRPC bridge - turns the pendant's blocking RPC call into a
# sensor_data_received event, then waits for the corresponding plan_executed
# event before returning.
# ---------------------------------------------------------------------------
class RobotBridge(Node):
    """Bridges the synchronous UR XMLRPC contract to the async event bus."""

    def __init__(self, config=None, verbose=True):
        super().__init__(config=config, verbose=verbose)
        self._name = "RobotBridge"
        self._done = threading.Event()

    def register_callbacks(self):
        self.register_event_callback(
            event_key="plan_executed",
            callback=lambda msg: self._done.set(),
        )

    def take_new_image_and_detect(self, tcp_pose, detection_type_string,
                                  screw_fixture=False, save_detections=True,
                                  region_of_interest_type_screw=True):
        """Called by the UR pendant. Blocks until Execute publishes plan_executed."""
        self._done.clear()
        payload = json.dumps({
            "detection_type": detection_type_string,
            "tcp_pose": tcp_pose,
        })
        self.publish_event(event_key="sensor_data_received", message=payload)
        if not self._done.wait(timeout=30.0):
            raise TimeoutError("MAPLE-K cycle did not complete in 30s")
        return None  # coords fetched via get_detected_object_coords

    def get_detected_object_coords(self, obj_type, camera_view="close"):
        """Called by the UR pendant to fetch the last-cycle detection coords."""
        # TODO: port calc_img_point_to_base_frame / get_coordinates_list from
        # the original screwSegmentation.py onto ScrewDetectionCore.
        raise NotImplementedError(
            "Coordinate transforms not yet ported onto ScrewDetectionCore."
        )


def main(robot_ip="192.168.1.100", xmlrpc_port=50000):
    logging.getLogger().setLevel(logging.INFO)
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
    )
    ss.set_core(core)

    # --- Config ------------------------------------------------------------
    config = {}
    try:
        import yaml
        cfg_path = os.path.join(os.path.dirname(__file__), "config.yaml")
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
    server = SimpleXMLRPCServer(("", xmlrpc_port), allow_none=True, logRequests=False)
    server.register_function(bridge.take_new_image_and_detect, "take_new_image_and_detect")
    server.register_function(bridge.get_detected_object_coords, "get_detected_object_coords")
    print(f"Listening on port {xmlrpc_port} ...")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("Server stopped.")


if __name__ == "__main__":
    main()
