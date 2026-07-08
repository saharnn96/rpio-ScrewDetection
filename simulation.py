"""Simulation entry point (composition root).

Wires the MANAGED system with SIMULATED adapters to the MANAGING system
(MAPLE-K nodes) and drives the loop with timed ticks:

    managing_system/   the five MAPLE-K Nodes + adaptation policy
    managed_system/    ScrewDetectionCore + Simulated{Camera,RTDE,Detector}

Run:    python simulation.py

Nothing here knows about YOLO, RealSense or the UR robot - that's the point
of the split. Swap the sim adapters for the real ones (see `real_main.py`)
to run the same MAPLE-K loop against real hardware.
"""

import json
import logging
import os
import time

from managed_system import core as ss
from managed_system.core import ScrewDetectionCore
from managed_system.adapters.sim import (
    SimulatedCamera, SimulatedRTDE, SimulatedDetector,
)
from managing_system.nodes import Node, build_nodes, run_dashboard, USING_REAL_RPCLPY
from managing_system.messages import ActionCommand

_HERE = os.path.dirname(os.path.abspath(__file__))


# ---------------------------------------------------------------------------
# Fake "sensor" that pumps the loop.
#
# In real deployment the UR robot's XMLRPC call plays this role - see
# `bridge.py` for the RobotBridge. In sim, we just emit sensor_data_received
# events on a fixed cadence.
# ---------------------------------------------------------------------------
class SensorPublisher(Node):
    def __init__(self, config=None, verbose=True):
        super().__init__(config=config, verbose=verbose)
        self._name = "Sensor"

    def emit(self, detection_type, tcp_pose):
        payload = json.dumps({"detection_type": detection_type,
                              "tcp_pose": tcp_pose})
        self.publish_event(event_key="sensor_data_received", message=payload)


def _load_managing_config():
    try:
        import yaml
        cfg_path = os.path.join(_HERE, "managing_system", "config.yaml")
        if os.path.exists(cfg_path):
            with open(cfg_path) as f:
                return yaml.safe_load(f) or {}
    except Exception as exc:
        print(f"(managing_system/config.yaml not loaded: {exc})")
    return {}


def main(num_ticks=24, interval=0.05):
    logging.getLogger().setLevel(logging.INFO)
    print("=" * 78)
    print(f"Screw-detection MAPLE-K simulation  "
          f"(rpclpy={'REAL' if USING_REAL_RPCLPY else 'local shim'})")
    print("=" * 78)

    # --- MANAGED system: simulated adapters + core --------------------------
    camera = SimulatedCamera(drift_after=12)
    rtde = SimulatedRTDE()
    detector = SimulatedDetector(model_id=1)
    core = ScrewDetectionCore(camera=camera, rtde=rtde, detector=detector,
                              out_dir="./_sim_out")
    ss.set_core(core)

    # --- MANAGING system: MAPLE-K nodes --------------------------------------
    config = _load_managing_config()
    # A short window makes the 24-tick demo trip the adaptation quickly.
    config.setdefault("Adaptation_Config", {})["entropy_window_size"] = 8
    build_nodes(config)
    sensor = SensorPublisher(config.get("Monitor_Config"))

    run_dashboard(host="127.0.0.1", port=8050, debug=False, start_trust=True)

    print("\nDriving sensor_data_received ticks (lighting goes dim at tick 12)...\n")
    for tick in range(num_ticks):
        print(f"\n---------- tick {tick:02d} ----------")
        sensor.emit(detection_type="pc_screen", tcp_pose=rtde.get_tcp_pose())
        time.sleep(interval)

    print("\n" + "=" * 78)
    print(f"Simulation finished. Final deployed model id = {ss.CORE.detector.model_id}")
    last_cmd = sensor.read_knowledge(ActionCommand)
    if last_cmd and last_cmd.command:
        print(f"Last action: {last_cmd.command} -> model {last_cmd.model_id} @ {last_cmd.timestamp}")
    print("=" * 78)


if __name__ == "__main__":
    main()
