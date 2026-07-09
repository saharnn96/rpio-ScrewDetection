# rpio-ScrewDetection — self-adaptive screw detection (MAPLE-K)

Port of the MAPLE-K loop that lives inside the original
`screw_detection/python/ScrewSegmentation.py` (the `take_new_image_and_detect`
method), restructured along the self-adaptive-systems reference architecture:
a **managed system** that does the work and a **managing system** that
watches and adapts it. The same MAPLE-K logic runs against a fake
camera/robot/model (laptop), the light box (real camera, mocked robot), or
the full field deployment — only the composition root changes.

The original code under `screw_detection/` is kept as a read-only parity
reference and is not modified.

---

## Architecture

```
   ┌──────────────────────────────────────────────────────┐
   │  MANAGING SYSTEM              managing_system/       │
   │  The five rpclpy Nodes: Monitor, Analysis, Plan,     │  nodes.py
   │  Legitimate, Execute. Knowledge classes (the K).     │  messages.py
   │  Adaptation policy (thresholds, window, candidate).  │  config.yaml
   │  Events/knowledge ride on rpclpy + redis (127.0.0.1).│
   └───────────────┬───────────────────▲──────────────────┘
                   │ probes: capture(), detect(),
                   │         append_rolling_image(), log_knowledge()
                   │ effectors: swap_model()
   ┌───────────────▼───────────────────┴──────────────────┐
   │  MANAGED SYSTEM               managed_system/        │
   │  ScrewDetectionCore (application primitives +        │  core.py
   │  coordinate transforms + the probe/effector surface).│
   │  Hardware adapters:                                  │  adapters/sim.py
   │    Simulated / Real / Lightbox                       │  adapters/real.py
   │  Deployment facts (robot IP, ports, model paths).    │  adapters/lightbox.py
   │                                                      │  config.yaml
   └──────────────────────────────────────────────────────┘

   Composition roots wire the two systems together:
     simulation.py     sim adapters, timed ticks           (no hardware)
     lightbox_main.py  real camera + mocked robot, ticks   (light box)
     real_main.py      real adapters + bridge.py           (field)

   bridge.py sits at the boundary: the UR pendant's synchronous XML-RPC
   contract (port 50000, same method surface as the original server) mapped
   onto the async event bus. It is neither an adapter (it is inbound, not
   outbound) nor a node of the loop (it exists only for the pendant).
```

**Debug rule of thumb:** adaptation logic bug → `managing_system/nodes.py`.
Wrong detection or coordinate math → `managed_system/core.py` or an adapter.
Camera / robot / model not talking → `managed_system/adapters/` only.
Pendant call failing → `bridge.py` (grep `xmlrpc_errors.log` for the
`[rpc:xxxxxxxx]` id shown on the pendant).

---

## Prerequisites (all modes)

1. Python 3.11 and `pip install -r requirements.txt`.
2. A **redis server on `127.0.0.1:6379`** — the MAPLE-K nodes, knowledge
   store and dashboard all run over it. Any recent redis works with zero
   configuration (`sudo apt install redis-server`, or
   `docker run -d -p 6379:6379 redis`). Keep every host set to `127.0.0.1`,
   not `localhost` — on Windows the IPv6 fallback adds ~21 s per connection.

---

## Running

### Simulation (no hardware)
```bash
python simulation.py
```
Around tick ~12 the simulated lighting drops and you should see, in order:

1. `ANALYZE` running-average entropy climbs above the `0.35` threshold.
2. `ANALYZE` emits `anomaly_detected`.
3. `PLAN` proposes candidate model id `2`.
4. `LEGITIMATE` re-tests model 2 on the rolling images, accepts the swap.
5. `EXECUTE` commits the swap; subsequent ticks show low entropy again.

Artefacts land in `./_sim_out/`.

### Light box (real camera, no robot)
```bash
pip install pyrealsense2                     # + ultralytics torch torchmetrics deepluq for real detection
python lightbox_main.py
```
Change the box lighting by hand mid-run to trigger the adaptation.

### Real hardware (field)

1. **Install the hardware extras** listed (commented) at the bottom of
   `requirements.txt`: `pyrealsense2`, `ultralytics`, `torch`,
   `torchmetrics`, `deepluq`, `ur_rtde`. Note: `run_uq` expects the
   project's **patched ultralytics fork** (the one exposing raw logits via
   `Results.detection`), same as the original code.
2. **Add the original repo**: place the untouched `screw_detection` tree in
   this repo's root — `<repo-root>/screw_detection/` (it is gitignored).
   The real adapters use it for three things:
   * the vendored `RDTEReceive` package (`screw_detection/python/`),
   * the hand-eye calibration
     (`screw_detection/python/camera_robot_calibration/screwdriver_tcp/CalibData/`),
   * the model weights.
3. **Configure** `managed_system/config.yaml`: set `robot_ip`, check
   `xmlrpc_port`, and point `detector.model_paths` ids 1 and 2 at your two
   lighting-calibrated closeup `.pt` files (the in-code defaults are
   fixture-weight stand-ins).
4. **Run**:
   ```bash
   python real_main.py
   ```
   This starts the MAPLE-K nodes, the dashboard, and the XML-RPC bridge,
   then serves the pendant on port 50000 — run the URScript program as
   before. The RPC surface is identical to the original
   (`take_new_image_and_detect`, `get_detected_object_coords`,
   exposure/calibration calls, …).

If something fails deep inside a detection cycle, the pendant call returns a
timeout Fault (default 120 s) instead of the original exception text — the
traceback is in the Analysis node's log on the dashboard, and RPC errors are
also written to `xmlrpc_errors.log` with a `[rpc:xxxxxxxx]` correlation id.

### Dashboard

Open **http://127.0.0.1:8050** while any entry point runs (it is started
automatically). *System Logs* shows each node's log stream — tick the
sources you want, e.g. `Analysis:logs` for entropy values and anomaly
warnings. The timeline at the top shows when each MAPLE-K phase executed.

### Tests
Plain Python scripts (no pytest); redis must be running. Hardware checks
SKIP cleanly when the device/stack is absent, so the same scripts run on a
laptop, on the light box and in the field:
```bash
python tests/test_lightbox_adapters.py     # step 1: adapter functions
python tests/test_lightbox_application.py  # step 2: ScrewDetectionCore
python tests/test_lightbox_maple_k.py      # step 3: full MAPLE-K loop
python tests/test_real_bridge.py           # step 4: pendant XML-RPC contract
python tests/test_real_adapters.py         # step 5: field-deployment checks
```

---

## Simulation notes

The sim adapters (`managed_system/adapters/sim.py`) satisfy the same
interfaces as the real ones. `SimulatedCamera` renders a synthetic scene
whose background dims after tick 12 — that brightness swing is the "lighting
change" driving the adaptation. `SimulatedDetector` computes screw entropy
as a pure function of frame brightness:

```
optimum = {1: 150.0, 2: 50.0}
entropy = clip(|brightness - optimum[model_id]| / 150, 0.02, 0.95)
```

so model 1 is confident on bright frames and model 2 on dim ones. Because
entropy is deterministic given the image, Legitimate's re-test on the saved
rolling frames is consistent with what Analyze measured live — mirroring how
the real LEGITIMATE region works — and after the drift model 2 strictly
beats model 1, so the swap is accepted.

The bus is never simulated: events and knowledge go through rpclpy + redis
in every mode. Event dispatch is asynchronous (pub/sub listener threads),
which is why the loop tests wait for events instead of assuming one emit
runs the whole chain inline.

---

## What the MAPLE-K loop does per cycle

| Phase | Node | Reads (knowledge) | Writes (knowledge) | Publishes |
|---|---|---|---|---|
| **M** Monitor | `Monitor.monitor` | `ActiveModel` | `FrameRef` | `observation_recorded` |
| **A** Analyze | `Analysis.analysis` | `FrameRef`, `ActiveModel`, `EntropyHistory`, `InPlanning` | `Detections`, `EntropyHistory`, `RunningAvgEntropy`, `InPlanning` | `detection_completed` (every cycle), `anomaly_detected` (only if avg > threshold + not already adapting) |
| **P** Plan | `Plan.planner` | `ActiveModel`, `ReplanningCounter` | `CandidateModel` | `plan_generated` |
| **L** Legitimate | `Legitimate.legitimizer` | `CandidateModel`, `RunningAvgEntropy`, rolling images on disk | `LegitResult`, `ReplanningCounter` | `plan_validated` on accept, `plan_rejected` on reject (with `max_replans` guard) |
| **E** Execute | `Execute.executer` | `LegitResult`, `FrameRef`, `Detections` | `ActionCommand`, `ActiveModel`, `EntropyHistory` (reset), `InPlanning=False` | `plan_executed` |
| **K** Knowledge | (shared) | — | `knowledge_log.csv` appended per swap | — |

The adaptation parameters each node uses (window, threshold, candidate id,
budget) come from `Adaptation_Config` in `managing_system/config.yaml`,
applied by `build_nodes()`.

The full event wiring (who publishes/consumes which event, topics, QoS)
lives in `managing_system/config.yaml` and is summarized in its header
comment.

---

## Remaining field-deployment gaps

| # | Task | Files to touch |
|---|---|---|
| 1 | Point ids 1/2 at the real closeup weights (in-code defaults are fixture stand-ins) | `managed_system/config.yaml` |
| 2 | Segmentation model (screen/screen_frame obj_types) not ported — those pendant calls raise a Fault | `managed_system/adapters/real.py`, `core.py` |
| 3 | Brightness-variant closeup models + auto-exposure from the original are not ported | `managed_system/adapters/real.py` |
| 4 | (Optional) smarter Plan policy with more than one candidate model | `managing_system/nodes.py` |
