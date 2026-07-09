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
     simulation.py     sim adapters, timed ticks           (offline demo)
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

## File map

| File | System | Role |
|---|---|---|
| `managing_system/nodes.py` | managing | The five `Node` subclasses + `build_nodes(config)` + `DEFAULT_ADAPTATION`. Re-exports `Node`, `timeit_callback`, `run_dashboard` from rpclpy and adds the non-blocking `start_dashboard()`. |
| `managing_system/messages.py` | managing | Plain-attribute knowledge classes (`FrameRef`, `Detections`, `EntropyHistory`, `RunningAvgEntropy`, `ActiveModel`, `CandidateModel`, `InPlanning`, `ReplanningCounter`, `LegitResult`, `ActionCommand`). Image data never goes through here — only file paths. |
| `managing_system/config.yaml` | managing | Node wiring (events/knowledge/topics/redis) + `Adaptation_Config` (entropy threshold, window size, candidate model id, replan budget) + `RobotBridge_Config`. |
| `managed_system/core.py` | managed | `ScrewDetectionCore`: probe/effector surface, detection primitives, image-point→base-frame coordinate transforms, `CORE` singleton set by the composition root. Domain types (`DetectionResult`, `DetectionClasses`). |
| `managed_system/adapters/sim.py` | managed | `SimulatedCamera` / `SimulatedRTDE` / `SimulatedDetector`. Zero hardware deps. |
| `managed_system/adapters/real.py` | managed | `RealCamera` / `RealRTDE` / `RealDetector` (YOLO + MC-dropout UQ + holder filtering). Lazy-imports pyrealsense2 / RDTEReceive / ultralytics / deepluq. |
| `managed_system/uq.py` | managed | `run_uq` — MC-dropout forward passes + entropy metrics (deepluq). Only consumer: `RealDetector`. Identical to the original `inference_uq_single_detection.py`. |
| `managed_system/adapters/lightbox.py` | managed | Real camera + mocked robot + real-detector-or-sim-fallback for the educational box. |
| `managed_system/config.yaml` | managed | Deployment facts: robot IP, XML-RPC port, camera settings, calibration folder, model weight paths. |
| `bridge.py` | boundary | `RobotBridge` + `LoggingXMLRPCServer` + `build_server()`: the UR pendant contract. |
| `simulation.py`, `lightbox_main.py`, `real_main.py` | entry | Composition roots. |
| `screw_detection/` | reference | The original untouched codebase (parity reference, calibration data, model weights). |
| `tests/` | — | Script-style checks, no pytest needed (see Running). |

---

## Running

### Offline simulation
```bash
pip install -r requirements.txt
python simulation.py
```
Around tick ~12 the simulated lighting drops and you should see, in order:

1. `ANALYZE` running-average entropy climbs above `0.5`.
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
1. Install the extras in `requirements.txt` (uncomment the `pyrealsense2`,
   `ultralytics`, `torch`, `deepluq`, `ur_rtde` block).
2. Point `detector.model_paths` in `managed_system/config.yaml` at your real
   closeup `.pt` files (the in-code defaults are fixture-weight stand-ins).
3. Check `robot_ip` / `xmlrpc_port` in `managed_system/config.yaml`.
4. Start a redis server on 127.0.0.1:6379 (the MAPLE-K nodes require it).
5. `python real_main.py` — then run the URScript program on the pendant.

### Tests
Plain Python scripts (no pytest); hardware checks SKIP cleanly when the
device/stack is absent, so the same scripts run on a laptop and on the box:
```bash
python tests/test_lightbox_adapters.py     # step 1: adapter functions
python tests/test_lightbox_application.py  # step 2: ScrewDetectionCore
python tests/test_lightbox_maple_k.py      # step 3: full MAPLE-K loop
python tests/test_real_bridge.py           # step 4: pendant XML-RPC contract
python tests/test_real_adapters.py         # step 5: field-deployment checks
```

---

## How each hardware dependency is simulated

Everything lives in `managed_system/adapters/sim.py`. Each class satisfies
the interface documented in `managed_system/core.py`.

### 1. Camera — `SimulatedCamera` (replaces `pyrealsense2`)
`get_color_image()` returns a synthetic 720×1280×3 BGR frame:
* Solid background at `bright_level=150` for the first `drift_after=12` ticks,
  then `dim_level=45`. That brightness swing is the "lighting change" that
  drives the whole MAPLE-K adaptation.
* A red rectangle (holder) + green circle (screw) so the saved frames look
  like a plausible scene.
* Small uniform noise so brightness isn't exactly constant.

Depth is omitted — the adaptation loop doesn't use it. (The real camera
snapshots aligned depth at capture time for the coordinate transforms.)

### 2. Robot — `SimulatedRTDE` (replaces `RTDEReceive` + XMLRPC)
`get_tcp_pose()` returns a fixed pose `{x: 0.4, y: 0, z: 0.5, rx: 0, ry: 3.14, rz: 0}`.
Tool current and runtime state are dropped (unused in the loop).

The pendant's blocking XMLRPC call is replaced in the sim by
`simulation.SensorPublisher.emit()`, which publishes a `sensor_data_received`
event carrying `{detection_type, tcp_pose}`. This is the same pattern
`bridge.RobotBridge` uses for real deployment — turning the sync RPC into an
event and blocking until Analysis publishes `detection_completed`.

### 3. YOLO + UQ — `SimulatedDetector` (replaces `ultralytics.YOLO` + `run_uq`)
`detect(image, model_id)` returns one holder + one screw `DetectionResult`
and a screw entropy computed as a **pure function of the frame's
brightness**:
```
optimum = {1: 150.0, 2: 50.0}
entropy = clip(|brightness - optimum[model_id]| / 150, 0.02, 0.95)
```

| brightness | model 1 entropy | model 2 entropy | interpretation |
|---|---|---|---|
| 150 (bright) | ~0.02 | ~0.66 | Model 1 is confident; the incumbent is fine. |
| 45  (dim)    | ~0.70 | ~0.03 | Model 1 loses confidence; model 2 is the right fit. |

Two properties fall out of this and matter for the trust check:
* **Deterministic given the image.** Legitimate re-reads the saved rolling
  frames from disk and re-runs the detector; because entropy is a pure
  function of the pixel values, its assessment is consistent with what
  Analyze measured live. This mirrors how the real LEGITIMATE region works.
* **Discriminates the two candidates.** After the drift, model 2 scores
  strictly lower entropy than model 1 on the same frames, so
  `candidate_avg < current_avg` and the swap is accepted.

### 4. The bus — rpclpy + redis (no offline shim)
The MAPLE-K nodes run on the real distributed bus in every mode, simulation
included: events and knowledge go through rpclpy over a redis server on
127.0.0.1:6379 (`127.0.0.1`, not `localhost` - the IPv6 fallback stalls ~21s
per connection on Windows). Event dispatch is asynchronous on pub/sub
listener threads, which is why the loop tests wait for events instead of
assuming one `sensor_data_received` emit runs the whole chain inline.

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

Interesting compared to the original: `Plan.planner` and the model-swap in
`Execute.executer` do something — in the original both were dead code.

### Event bus

| Event Key | Published By | Consumed By | Meaning |
|---|---|---|---|
| `sensor_data_received` | SensorPublisher tick or RobotBridge RPC | Monitor | New sensor reading arrived |
| `observation_recorded` | Monitor | Analysis | Observation stored, ready for anomaly check |
| `detection_completed` | Analysis | RobotBridge | This cycle's Detections are in the knowledge store |
| `anomaly_detected` | Analysis | Plan | Detection window exceeded threshold |
| `plan_generated` | Plan | Legitimate | New plan/model is ready for validation |
| `plan_rejected` | Legitimate | Plan | Validation failed, re-planning required |
| `plan_validated` | Legitimate | Execute | Plan passed validation, safe to deploy |
| `plan_executed` | Execute | (observers) | Plan deployed, monitoring resumed |

---

## Remaining field-deployment gaps

| # | Task | Files to touch |
|---|---|---|
| 1 | Point ids 1/2 at the real closeup weights (in-code defaults are fixture stand-ins) | `managed_system/config.yaml` |
| 2 | Segmentation model (screen/screen_frame obj_types) not ported — those pendant calls raise a Fault | `managed_system/adapters/real.py`, `core.py` |
| 3 | Brightness-variant closeup models + auto-exposure from the original are not ported | `managed_system/adapters/real.py` |
| 4 | (Optional) smarter Plan policy with more than one candidate model | `managing_system/nodes.py` |
