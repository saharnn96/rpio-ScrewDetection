"""Real-hardware entry point (composition root).

Wires the MANAGED system (real adapters + core) to the MANAGING system
(MAPLE-K nodes) and serves the UR pendant through the XML-RPC bridge:

    managing_system/   the five MAPLE-K Nodes + adaptation policy
                       (managing_system/config.yaml)
    managed_system/    ScrewDetectionCore + RealCamera/RealRTDE/RealDetector
                       (managed_system/config.yaml: robot IP, ports, models)
    bridge.py          pendant XML-RPC contract on port 50000, mapping the
                       synchronous UR calls onto the async event bus

To run in the field:
    pip install -r requirements.txt   # + uncomment the real-hardware extras
    python real_main.py
"""

import logging
import os

from managed_system import core as dc
from managed_system.core import ScrewDetectionCore
from managing_system.nodes import build_nodes, start_dashboard
from bridge import RobotBridge, build_server

_HERE = os.path.dirname(os.path.abspath(__file__))

# Fallback when managed_system/config.yaml is absent - same layout the
# original code used.
DEFAULT_CALIB_FOLDER = os.path.join(
    _HERE, "screw_detection", "python", "camera_robot_calibration",
    "screwdriver_tcp", "CalibData",
)


def _load_yaml(path):
    try:
        import yaml
        if os.path.exists(path):
            with open(path) as f:
                return yaml.safe_load(f) or {}
    except Exception as exc:
        print(f"({os.path.basename(path)} not loaded: {exc})")
    return {}


def main():
    logging.getLogger().setLevel(logging.INFO)
    print("=" * 78)
    print("Screw-detection MAPLE-K (real hardware)  (rpclpy + redis)")
    print("=" * 78)

    managed_cfg = _load_yaml(os.path.join(_HERE, "managed_system", "config.yaml"))
    managing_cfg = _load_yaml(os.path.join(_HERE, "managing_system", "config.yaml"))

    # --- MANAGED system: real adapters + core ------------------------------
    from managed_system.adapters.real import RealCamera, RealRTDE, RealDetector
    cam_cfg = managed_cfg.get("camera") or {}
    camera = RealCamera(width=cam_cfg.get("width", 1280),
                        height=cam_cfg.get("height", 720),
                        exposure_us=cam_cfg.get("exposure_us", 2500))
    rtde = RealRTDE(robot_ip=managed_cfg.get("robot_ip", "192.168.1.100"))
    det_cfg = managed_cfg.get("detector") or {}
    detector = RealDetector(model_id=det_cfg.get("model_id", 1),
                            model_paths=det_cfg.get("model_paths"))

    calib_folder = managed_cfg.get("calib_folder", DEFAULT_CALIB_FOLDER)
    if not os.path.isabs(calib_folder):
        calib_folder = os.path.join(_HERE, calib_folder)
    core = ScrewDetectionCore(
        camera=camera, rtde=rtde, detector=detector,
        out_dir=managed_cfg.get("out_dir", "./_prod_out"),
        calib_folder=calib_folder,
    )
    dc.set_core(core)

    # --- MANAGING system: MAPLE-K nodes + bridge ----------------------------
    build_nodes(managing_cfg)
    bridge = RobotBridge(managing_cfg.get("RobotBridge_Config"))
    bridge.register_callbacks()
    bridge.start()

    start_dashboard(host="127.0.0.1", port=8050, debug=False, start_trust=True)

    # --- Serve the pendant ---------------------------------------------------
    xmlrpc_port = managed_cfg.get("xmlrpc_port", 50000)
    server = build_server(bridge, xmlrpc_port)
    print(f"Listening on port {xmlrpc_port} ...")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("Server stopped.")


if __name__ == "__main__":
    main()
