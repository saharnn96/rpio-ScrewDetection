# maple_k_sim — offline MAPLE-K simulation for screw detection

This folder is a self-contained port of the MAPLE-K loop that lives inside
`python/screwSegmentation.py` (the `take_new_image_and_detect` method). It:

* Restructures the loop into five discrete `rpclpy` nodes.
* Provides drop-in fakes for the three hardware dependencies (RealSense camera,
  UR robot, YOLO + UQ pipeline) so the loop runs on a laptop with no hardware.
* Ships a fallback in-process event bus so the demo runs without installing
  `rpclpy` or standing up redis.
* Fills in the model-swap logic that is currently `pass` in the original code,
  so the full **Monitor → Analyze → Plan → Legitimate → Execute** chain
  actually fires end-to-end.

The main repo is not modified.

---

## File map

| File | Role |
|---|---|
| `messages.py` | Plain-attribute classes that flow through the knowledge store (`FrameRef`, `EntropyHistory`, `RunningAvgEntropy`, `ActiveModel`, `CandidateModel`, `InPlanning`, `ReplanningCounter`, `LegitResult`, `ActionCommand`). Keep them JSON-friendly — image data never travels through the store, only file paths. |
| `local_bus.py` | Fallback `Node` / `timeit_callback` / `run_dashboard` that mimic the rpclpy API in-process. Used automatically when `rpclpy` is not installed. |
| `screwSegmentation_rpio.py` | The port itself: `ScrewDetectionCore` (hardware-agnostic) plus the five `Node` subclasses (`Monitor`, `Analysis`, `Plan`, `Legitimate`, `Execute`). |
| `simulation.py` | `SimulatedCamera`, `SimulatedRTDE`, `SimulatedDetector`, and the driver `main()` that ticks the loop. Entry point for the demo. |
| `config.yaml` | Adapted MAPLE-K config (topics, QoS, redis endpoints). The offline shim ignores most of it; kept intact so the same code runs against real rpclpy + redis with zero changes. |

---

## Running the demo

```bash
cd maple_k_sim
python simulation.py
```

Prerequisites: `numpy`, `opencv-python`, `pyyaml`. No CUDA, no realsense, no
robot, no rpclpy, no redis.

The driver ticks the loop 24 times. Around tick ~12 the simulated lighting
drops and you should see, in order:

1. `ANALYZE` running-average entropy climbs above `0.5`.
2. `ANALYZE` emits `anomaly`.
3. `PLAN` proposes candidate model id `2`.
4. `LEGITIMATE` re-tests model 2 on the rolling images, accepts the swap.
5. `EXECUTE` commits the swap; subsequent ticks show low entropy again.

Artefacts land in `./_sim_out/` (rolling images, capture snapshots,
`knowledge_log.csv`).

---

## How each hardware dependency is simulated

The original code talks to three physical systems. Each is replaced by a small
adapter with the same interface the `ScrewDetectionCore` calls into.

### 1. Camera — `SimulatedCamera` (replaces `pyrealsense2`)

The real code opens a RealSense pipeline, waits for aligned color+depth
frames, and pulls per-pixel depth. The simulator only cares about producing a
BGR image whose brightness drives the MAPLE-K decision — depth is not part of
the adaptation loop, so it is omitted.

`get_color_image()` returns a synthetic `720×1280×3 uint8` frame:

* Solid background at `bright_level=150` for the first `drift_after=12` ticks,
  then `dim_level=45` for the rest — this is the "lighting change" that
  triggers the anomaly.
* A red rectangle drawn at the frame centre plays the role of a *holder*.
* A green circle inside it plays the role of a *screw*.
* Small uniform noise is added so brightness is not exactly constant.

Nothing about the camera is asynchronous; each call advances an internal tick
counter, which is how the "drift" is scheduled.

### 2. Robot — `SimulatedRTDE` (replaces `RTDEReceive` / the UR pendant)

The real code has two robot interfaces:

* A continuous **RTDE** telemetry stream (TCP pose, tool current, runtime state).
* An inbound **XMLRPC** command channel where the pendant calls
  `take_new_image_and_detect(tcp_pose, detection_type)` and blocks on the
  return.

The simulator replaces both:

* `SimulatedRTDE.get_tcp_pose()` returns a **fixed** TCP pose. Tool current
  and runtime state are not used inside the MAPLE-K loop, so they are dropped.
* The blocking XMLRPC call is replaced by `SensorPublisher.emit()`, which
  publishes a `SensorData` event carrying `{detection_type, tcp_pose}`. This is
  the pattern I recommended in the earlier bottleneck discussion: turn the
  synchronous RPC into an event, then let the nodes react.

The pose is fixed because the original TCP pose only affects the
image-to-base-frame coordinate transform (`calc_img_point_to_base_frame`),
which is downstream of the adaptation loop. Getting the loop itself to fire
does not need robot kinematics.

### 3. YOLO + UQ — `SimulatedDetector` (replaces `ultralytics.YOLO` + `run_uq`)

This is the most important fake. The real pipeline runs YOLO with MC-Dropout
`T=10` times per frame, clusters detections with WBF, and computes classifi-
cation entropy per detection. It depends on `deepluq`, on a custom YOLO fork
whose predictions carry a `.detection` attribute, and on `.pt` weight files at
hardcoded Linux paths. None of those are available in a sim.

`SimulatedDetector.detect(image, model_id)` returns:

* One `HOLDER` and one `SCREW` `DetectionResult` centred on the frame — enough
  for the `filter_out_detections_not_within_holders` logic and the knowledge
  log to have real data.
* A screw entropy computed as:
  ```
  brightness = grayscale_mean(image)
  entropy    = clip(|brightness - optimum[model_id]| / 150, 0.02, 0.95)
  optimum = {1: 150.0, 2: 50.0}
  ```

That formula is the whole trick that makes the demo work:

| brightness | model 1 entropy | model 2 entropy | interpretation |
|---|---|---|---|
| 150 (bright) | ~0.02 | ~0.66 | Model 1 is confident; the incumbent is fine. |
| 45  (dim)    | ~0.70 | ~0.03 | Model 1 loses confidence; model 2 is the right fit. |

Two properties fall out of this that matter for the trustworthiness check:

* **Deterministic given the image.** Legitimate re-reads the saved rolling
  frames from disk and re-runs the detector; because entropy is a pure
  function of the pixel values, its assessment is consistent with what
  Analyze measured live. This mirrors how the real LEGITIMATE region works —
  it reads the same rolling-average JPEGs back and re-scores them.
* **Discriminates the two candidates.** After the lighting drift, model 2
  scores strictly lower entropy than model 1 on the same frames, so
  `candidate_avg < current_avg` and the swap is accepted.

`model_id` is a plain integer that Execute mutates on a successful swap.

### 4. rpclpy — `local_bus.py` (replaces the real distributed bus)

Not hardware, but same idea: an interface swap. `local_bus.Node` provides
`write_knowledge` / `read_knowledge` / `publish_event` /
`register_event_callback` / `start` — signatures identical to
`rpclpy.node.Node`. Two behaviour differences:

* Publish is **synchronous**: one `SensorData` emit drives the entire chain
  Monitor → Analyze → (Plan → Legitimate → Execute) on the same call stack
  before returning. In real rpclpy the same chain runs across processes via
  redis pub/sub. Per-node logic is identical either way.
* The knowledge store is a plain dict keyed by class name, not a redis db.

`screwSegmentation_rpio.py` picks up real rpclpy if present, or the shim if
not — nothing else changes.

---

## What the MAPLE-K loop does in this simulation

| Phase | Node | Reads (knowledge) | Writes (knowledge) | Publishes |
|---|---|---|---|---|
| **M** Monitor | `Monitor.monitor` | `ActiveModel` | `FrameRef` (path, brightness, TCP pose) | `new_data` |
| **A** Analyze | `Analysis.analysis` | `FrameRef`, `ActiveModel`, `EntropyHistory`, `InPlanning` | `Detections`, `EntropyHistory`, `RunningAvgEntropy`, `InPlanning` | `anomaly` (only if avg > 0.5 and not already adapting) |
| **P** Plan | `Plan.planner` | `ActiveModel`, `ReplanningCounter` | `CandidateModel` | `new_plan` |
| **L** Legitimate | `Legitimate.legitimizer` | `CandidateModel`, `RunningAvgEntropy`, rolling images on disk | `LegitResult`, `ReplanningCounter` | `isLegit` on accept, `anomaly` on reject (with `max_replans` guard) |
| **E** Execute | `Execute.executer` | `LegitResult`, `FrameRef`, `Detections` | `ActionCommand`, `ActiveModel`, `EntropyHistory` (reset), `InPlanning` (false) | `action_command` |
| **K** Knowledge | (shared) | — | `knowledge_log.csv` appended per swap | — |

The interesting bit versus the original code: `Plan.planner` and the
model-swap in `Execute.executer` **do something** — in the original they are
still `pass`.

---

## Porting to real hardware / real rpclpy

The core has three injection points. To go live, replace each with a real
adapter and drop the sim fakes:

| Interface | Sim class | Real adapter to build |
|---|---|---|
| `camera.get_color_image()` | `SimulatedCamera` | Thin wrapper around the existing `aligned_frames_and_images` from `python/screwSegmentation.py`. |
| `rtde.get_tcp_pose()` | `SimulatedRTDE` | Wrapper around `RTDEReceive.receive_data()['actual_TCP_pose']`. |
| `detector.detect(image, model_id)` | `SimulatedDetector` | Owns YOLO instances keyed by `model_id`; calls `run_uq(...)` + `uq_analysis(...)` and returns the same `(list[DetectionResult], screw_entropy)` tuple. |

Then either:

* Run `python simulation.py` after swapping `SimulatedCamera/RTDE/Detector`
  for the real ones (fastest path — reuses the driver and the shim).
* Install `rpclpy` + start redis and run `python screwSegmentation_rpio.py`
  once you have wired the `main()` at the bottom of that file to construct
  the real adapters. The nodes themselves need no change.

The synchronous XMLRPC contract with the UR pendant is the one thing the port
does **not** solve — see the earlier "Bottleneck 1" discussion. In the current
sim, `SensorPublisher.emit()` stands in for that call; real deployment needs
an XMLRPC → event-bus bridge that publishes `SensorData` and then blocks
waiting for the corresponding `ActionCommand` before returning to the robot.
