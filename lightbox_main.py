"""Light-box entry point (composition root).

Same wiring as `simulation.py` / `real_main.py`, but for the educational
light-box setup: REAL RealSense camera, MOCKED robot, real detector when the
ML stack is installed.

    managing_system/   the five MAPLE-K Nodes + adaptation policy
    managed_system/    ScrewDetectionCore + LightboxCamera (real) /
                       LightboxRTDE (mock) / RealDetector or sim fallback

Run:    python lightbox_main.py

There is no UR pendant to trigger cycles, so like the simulation we pump
`sensor_data_received` ticks on a fixed cadence. Change the light-box
brightness by hand mid-run to drive the MAPLE-K adaptation (entropy rises ->
anomaly -> Plan/Legitimate/Execute swap the active model).

Ctrl+C stops the loop and releases the camera.
"""

import logging
import os
import time

from managed_system import core as ss
from managed_system.core import ScrewDetectionCore
from managed_system.adapters.lightbox import (
    LightboxCamera, LightboxRTDE, build_lightbox_detector,
)
from managing_system.nodes import build_nodes, run_dashboard, USING_REAL_RPCLPY
from managing_system.messages import ActiveModel, RunningAvgEntropy
from simulation import SensorPublisher, _load_managing_config

_HERE = os.path.dirname(os.path.abspath(__file__))


def main(num_ticks=None, interval=2.0, detection_type="pc_screen"):
    """`num_ticks=None` runs until Ctrl+C."""
    logging.getLogger().setLevel(logging.INFO)
    print("=" * 78)
    print(f"Screw-detection MAPLE-K light-box run  "
          f"(rpclpy={'REAL' if USING_REAL_RPCLPY else 'local shim'})")
    print("=" * 78)

    # --- MANAGED system: lightbox adapters + core ----------------------------
    camera = LightboxCamera(width=1280, height=720, exposure_us=2500)
    rtde = LightboxRTDE()
    detector = build_lightbox_detector(model_id=1)

    try:
        core = ScrewDetectionCore(camera=camera, rtde=rtde, detector=detector,
                                  out_dir="./_lightbox_out")
        ss.set_core(core)

        # --- MANAGING system: MAPLE-K nodes + tick source --------------------
        config = _load_managing_config()
        # A short window reacts fast to hand-driven lighting changes.
        config.setdefault("Adaptation_Config", {})["entropy_window_size"] = 8
        build_nodes(config)
        sensor = SensorPublisher(config.get("Monitor_Config"))

        run_dashboard(host="127.0.0.1", port=8050, debug=False, start_trust=True)

        print("\nPumping sensor_data_received ticks from the light box.")
        print("Tip: change the box lighting mid-run to trigger the MAPLE-K "
              "model swap.\n")
        tick = 0
        try:
            while num_ticks is None or tick < num_ticks:
                print(f"\n---------- tick {tick:03d} ----------")
                sensor.emit(detection_type=detection_type,
                            tcp_pose=rtde.get_tcp_pose())
                avg = sensor.read_knowledge(RunningAvgEntropy)
                active = sensor.read_knowledge(ActiveModel)
                avg_s = f"{avg.value:.3f}" if avg and avg.value is not None else "n/a"
                print(f"active_model={active.model_id if active else '?'}  "
                      f"running_avg_entropy={avg_s}")
                tick += 1
                time.sleep(interval)
        except KeyboardInterrupt:
            print("\nStopped by user.")

        print("\n" + "=" * 78)
        print(f"Light-box run finished after {tick} ticks. "
              f"Final deployed model id = {ss.CORE.detector.model_id}")
        print("=" * 78)
    finally:
        camera.close()


if __name__ == "__main__":
    main()
