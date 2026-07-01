"""Simulation entry point.

Wires the three layers with SIMULATED adapters and drives the MAPLE-K loop:

    Layer 3   maple_k          <-- the five Nodes
    Layer 2   screwSegmentation <-- ScrewDetectionCore
    Layer 1   sim_adapters     <-- SimulatedCamera / SimulatedRTDE / SimulatedDetector

Run:    python simulation.py

Nothing here knows about YOLO, RealSense or the UR robot - that's the point
of the layering. Swap `sim_adapters` for `real_adapters` (see `real_main.py`)
to get the same MAPLE-K loop against real hardware.
"""

import json
import logging
import os
import time

import screwSegmentation as ss
from screwSegmentation import ScrewDetectionCore
from maple_k import Node, build_nodes, run_dashboard, USING_REAL_RPCLPY
from sim_adapters import SimulatedCamera, SimulatedRTDE, SimulatedDetector
from messages import ActionCommand


# ---------------------------------------------------------------------------
# Fake "sensor" that pumps the loop.
#
# In real deployment the UR robot's XMLRPC call plays this role - see
# `real_main.py` for the bridge. In sim, we just emit SensorData events on a
# fixed cadence.
# ---------------------------------------------------------------------------
class SensorPublisher(Node):
    def __init__(self, config=None, verbose=True):
        super().__init__(config=config, verbose=verbose)
        self._name = "Sensor"

    def emit(self, detection_type, tcp_pose):
        payload = json.dumps({"detection_type": detection_type,
                              "tcp_pose": tcp_pose})
        self.publish_event(event_key="SensorData", message=payload)


def main(num_ticks=24, interval=0.05):
    logging.getLogger().setLevel(logging.INFO)
    print("=" * 78)
    print(f"Screw-detection MAPLE-K simulation  "
          f"(rpclpy={'REAL' if USING_REAL_RPCLPY else 'local shim'})")
    print("=" * 78)

    # --- Layer 1: simulated adapters --------------------------------------
    camera = SimulatedCamera(drift_after=12)
    rtde = SimulatedRTDE()
    detector = SimulatedDetector(model_id=1)

    # --- Layer 2: core + config -------------------------------------------
    core = ScrewDetectionCore(
        camera=camera, rtde=rtde, detector=detector,
        out_dir="./_sim_out", entropy_window_size=8, entropy_threshold=0.5,
        candidate_model_id=2, max_replans=3,
    )
    ss.set_core(core)

    config = {}
    try:
        import yaml
        cfg_path = os.path.join(os.path.dirname(__file__), "config.yaml")
        if os.path.exists(cfg_path):
            with open(cfg_path) as f:
                config = yaml.safe_load(f) or {}
    except Exception as exc:
        print(f"(config.yaml not loaded: {exc})")

    # --- Layer 3: MAPLE-K nodes -------------------------------------------
    build_nodes(config)
    sensor = SensorPublisher(config.get("Monitor_Config") if config else None)

    run_dashboard(host="127.0.0.1", port=8050, debug=False, start_trust=True)

    print("\nDriving SensorData ticks (lighting goes dim at tick 12)...\n")
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
