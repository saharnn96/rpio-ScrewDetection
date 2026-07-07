"""Light-box entry point.

Same wiring as `simulation.py` / `real_main.py`, but for the educational
light-box setup: REAL RealSense camera, MOCKED robot, real detector when the
ML stack is installed.

    Layer 3   maple_k            <-- the five Nodes          (unchanged)
    Layer 2   detection_core     <-- ScrewDetectionCore      (unchanged)
    Layer 1   lightbox_adapters  <-- LightboxCamera (real) /
                                     LightboxRTDE (mock) /
                                     RealDetector or sim fallback

Run:    python lightbox_main.py

There is no UR pendant to trigger cycles, so like the simulation we pump
`sensor_data_received` ticks on a fixed cadence. Change the light-box
brightness by hand mid-run to drive the MAPLE-K adaptation (entropy rises ->
anomaly -> Plan/Legitimate/Execute swap the active model).

Ctrl+C stops the loop and releases the camera.
"""

import os
import logging
import time

import detection_core as ss
from detection_core import ScrewDetectionCore
from maple_k import build_nodes, run_dashboard, USING_REAL_RPCLPY
from messages import ActiveModel, RunningAvgEntropy
from lightbox_adapters import LightboxCamera, LightboxRTDE, build_lightbox_detector
from simulation import SensorPublisher

_HERE = os.path.dirname(os.path.abspath(__file__))


def main(num_ticks=None, interval=2.0, detection_type="pc_screen"):
    """`num_ticks=None` runs until Ctrl+C."""
    logging.getLogger().setLevel(logging.INFO)
    print("=" * 78)
    print(f"Screw-detection MAPLE-K light-box run  "
          f"(rpclpy={'REAL' if USING_REAL_RPCLPY else 'local shim'})")
    print("=" * 78)

    # --- Layer 1: lightbox adapters ----------------------------------------
    camera = LightboxCamera(width=1280, height=720, exposure_us=2500)
    rtde = LightboxRTDE()
    detector = build_lightbox_detector(model_id=1)

    try:
        # --- Layer 2: core --------------------------------------------------
        core = ScrewDetectionCore(
            camera=camera, rtde=rtde, detector=detector,
            out_dir="./_lightbox_out",
            entropy_window_size=8, entropy_threshold=0.5,
            candidate_model_id=2, max_replans=3,
        )
        ss.set_core(core)

        # --- Config ---------------------------------------------------------
        config = {}
        try:
            import yaml
            cfg_path = os.path.join(_HERE, "config.yaml")
            if os.path.exists(cfg_path):
                with open(cfg_path) as f:
                    config = yaml.safe_load(f) or {}
        except Exception as exc:
            print(f"(config.yaml not loaded: {exc})")

        # --- Layer 3: MAPLE-K nodes + tick source ---------------------------
        build_nodes(config)
        sensor = SensorPublisher(config.get("Monitor_Config") if config else None)

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
