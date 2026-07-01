# maple_k_sim — offline MAPLE-K simulation for screw detection

Self-contained port of the MAPLE-K loop that lives inside
`python/screwSegmentation.py` (the `take_new_image_and_detect` method),
restructured into a clean **three-layer architecture** so it can run both
against a fake camera/robot/model (for debugging the loop on a laptop) and
against the real hardware (for field deployment) with zero changes to the
MAPLE-K logic itself.

The main repo is not modified.

---

## The three layers

```
   ┌──────────────────────────────────────────────────────┐
   │  Layer 3:  MAPLE-K loop                              │  maple_k.py
   │  The five rpclpy Nodes: Monitor, Analysis, Plan,     │
   │  Legitimate, Execute. Read/write knowledge.          │
   │  Publish events. Call into the core. No hardware.    │
   ├──────────────────────────────────────────────────────┤
   │  Layer 2:  Application (screwSegmentation)           │  screwSegmentation.py
   │  ScrewDetectionCore + adapter interfaces + domain    │
   │  types (DetectionResult, DetectionClasses,           │
   │  DetectionModels). Hardware-agnostic.                │
   ├──────────────────────────────────────────────────────┤
   │  Layer 1:  Adapters (hardware)                       │  sim_adapters.py
   │  SimulatedCamera / SimulatedRTDE / SimulatedDetector │  real_adapters.py
   │  RealCamera     / RealRTDE     / RealDetector        │
   │  Only place that touches pyrealsense2, ur_rtde,      │
   │  ultralytics, deepluq, ...                           │
   └──────────────────────────────────────────────────────┘

   Entry points wire the layers:
     simulation.py  =>  sim_adapters   (offline demo)
     real_main.py   =>  real_adapters  (field deployment + XMLRPC bridge)
```

**Debug rule of thumb:** MAPLE-K logic bug → Layer 3. Wrong detection or
brightness math → Layer 2 or the adapter. Camera / robot / model not talking
→ Layer 1 only.

---

## File map

| File | Layer | Role |
|---|---|---|
| `messages.py` | shared | Plain-attribute classes that flow through the knowledge store (`FrameRef`, `EntropyHistory`, `RunningAvgEntropy`, `ActiveModel`, `CandidateModel`, `InPlanning`, `ReplanningCounter`, `LegitResult`, `ActionCommand`). Image data never goes through here — only file paths. |
| `local_bus.py` | infra | Fallback for `rpclpy.node.Node` / `timeit_callback` / `run_dashboard` when rpclpy is not installed. In-process, synchronous. Used automatically. |
| `maple_k.py` | **Layer 3** | The five `Node` subclasses + `build_nodes(config)`. Re-exports `Node`, `timeit_callback`, `run_dashboard` from real rpclpy if available. |
| `screwSegmentation.py` | **Layer 2** | `ScrewDetectionCore` (application primitives) + `DetectionResult` / `DetectionClasses` / `DetectionModels` + the module-level `CORE` singleton set by the entry point. |
| `sim_adapters.py` | **Layer 1** | `SimulatedCamera`, `SimulatedRTDE`, `SimulatedDetector`. Zero hardware deps. |
| `real_adapters.py` | **Layer 1** | `RealCamera`, `RealRTDE`, `RealDetector`. Lazy-imports pyrealsense2 / RTDEReceive / ultralytics / deepluq inside the classes. |
| `simulation.py` | entry | Wires `sim_adapters` + core + nodes, drives 24 SensorData ticks. |
| `real_main.py` | entry | Wires `real_adapters` + core + nodes + `RobotBridge` (XMLRPC server that bridges the UR pendant's blocking call to the event bus). |
| `config.yaml` | shared | rpclpy config (topics, QoS, redis endpoints). Offline shim ignores most of it; kept intact so the same nodes run distributed. |
| `requirements.txt` | shared | Minimal offline deps; real-hardware deps commented for the field. |

---

## Running

### Offline simulation
```bash
cd maple_k_sim
pip install -r requirements.txt
python simulation.py
```
Around tick ~12 the simulated lighting drops and you should see, in order:

1. `ANALYZE` running-average entropy climbs above `0.5`.
2. `ANALYZE` emits `anomaly`.
3. `PLAN` proposes candidate model id `2`.
4. `LEGITIMATE` re-tests model 2 on the rolling images, accepts the swap.
5. `EXECUTE` commits the swap; subsequent ticks show low entropy again.

Artefacts land in `./_sim_out/`.

### Real hardware (field)
1. Install the extras in `requirements.txt` (uncomment the `pyrealsense2`,
   `ultralytics`, `torch`, `deepluq`, `ur_rtde` block).
2. Point `RealDetector.MODEL_PATHS` at your `.pt` files (`real_adapters.py`).
3. Finish the XMLRPC bridge in `real_main.py` — `get_detected_object_coords`
   is a stub; port the coordinate transforms from the original
   `python/screwSegmentation.py` onto `ScrewDetectionCore`.
4. Optionally `pip install rpclpy redis` and start a redis server if you
   want the nodes distributed instead of in-process.
5. Run:
   ```bash
   python real_main.py
   ```

---

## How each hardware dependency is simulated

Everything lives in `sim_adapters.py`. Each class satisfies the interface
documented in `screwSegmentation.py`.

### 1. Camera — `SimulatedCamera` (replaces `pyrealsense2`)
`get_color_image()` returns a synthetic 720×1280×3 BGR frame:
* Solid background at `bright_level=150` for the first `drift_after=12` ticks,
  then `dim_level=45`. That brightness swing is the "lighting change" that
  drives the whole MAPLE-K adaptation.
* A red rectangle (holder) + green circle (screw) so the saved frames look
  like a plausible scene.
* Small uniform noise so brightness isn't exactly constant.

Depth is omitted — the adaptation loop doesn't use it.

### 2. Robot — `SimulatedRTDE` (replaces `RTDEReceive` + XMLRPC)
`get_tcp_pose()` returns a fixed pose `{x: 0.4, y: 0, z: 0.5, rx: 0, ry: 3.14, rz: 0}`.
Tool current and runtime state are dropped (unused in the loop).

The pendant's blocking XMLRPC call is replaced in the sim by
`simulation.SensorPublisher.emit()`, which publishes a `SensorData` event
carrying `{detection_type, tcp_pose}`. This is the same pattern
`real_main.RobotBridge` uses for real deployment — turning the sync RPC
into an event and blocking on the corresponding `action_command`.

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

### 4. rpclpy — `local_bus.py` (replaces the real distributed bus)
Not hardware, but same idea: an interface swap. `local_bus.Node` provides
`write_knowledge` / `read_knowledge` / `publish_event` /
`register_event_callback` / `start` with identical signatures. Publish is
synchronous (one `SensorData` emit drives the whole chain on one stack).
`maple_k.py` imports real rpclpy if present, this shim otherwise.

---

## What the MAPLE-K loop does per cycle

| Phase | Node | Reads (knowledge) | Writes (knowledge) | Publishes |
|---|---|---|---|---|
| **M** Monitor | `Monitor.monitor` | `ActiveModel` | `FrameRef` | `new_data` |
| **A** Analyze | `Analysis.analysis` | `FrameRef`, `ActiveModel`, `EntropyHistory`, `InPlanning` | `Detections`, `EntropyHistory`, `RunningAvgEntropy`, `InPlanning` | `anomaly` (only if avg > threshold + not already adapting) |
| **P** Plan | `Plan.planner` | `ActiveModel`, `ReplanningCounter` | `CandidateModel` | `new_plan` |
| **L** Legitimate | `Legitimate.legitimizer` | `CandidateModel`, `RunningAvgEntropy`, rolling images on disk | `LegitResult`, `ReplanningCounter` | `isLegit` on accept, `anomaly` on reject (with `max_replans` guard) |
| **E** Execute | `Execute.executer` | `LegitResult`, `FrameRef`, `Detections` | `ActionCommand`, `ActiveModel`, `EntropyHistory` (reset), `InPlanning=False` | `action_command` |
| **K** Knowledge | (shared) | — | `knowledge_log.csv` appended per swap | — |

Interesting compared to the original: `Plan.planner` and the model-swap in
`Execute.executer` do something — in the original both are `pass`.

---

## Migration checklist (sim → field)

| # | Task | Files to touch |
|---|---|---|
| 1 | Point `RealDetector.MODEL_PATHS` at real `.pt` files | `real_adapters.py` |
| 2 | Port `calc_img_point_to_base_frame` + `get_coordinates_list` from the original onto `ScrewDetectionCore` | `screwSegmentation.py` |
| 3 | Finish `RobotBridge.get_detected_object_coords` to call the ported transforms | `real_main.py` |
| 4 | (Optional) `pip install rpclpy redis` + start redis for distributed nodes | none |
| 5 | Fill in a smarter Plan policy if you want more than one candidate model | `maple_k.py` |

**Untouched by field deployment:** all of `maple_k.py`, all of `messages.py`,
all of `screwSegmentation.py`'s core primitives, all of `config.yaml`. The
whole point of the layering.
