# Trustworthiness specification for the MAPLE-K loop

This directory holds the **runtime-verification specification** for this
system: the properties a trustworthiness checker must verify **before,
during and after** each adaptation episode. The monitor itself is developed
by the RV team; this is the contract they implement against.

* [`maple_k.lola`](maple_k.lola) — the specification in LOLA (stream
  equations). Each property is a `v_*` (violation) or `w_*` (warning)
  output stream; `violation` / `warning` aggregate them.
* This README — how to derive the input streams from the running system,
  the tick model, and the property catalogue in prose.

The LOLA file uses only the common core of the language (`in`/`out`
declarations, bounded past references `s[-1, default]`, `if-then-else`,
boolean/arithmetic operators), so it should translate 1:1 to the monitor's
dialect if the parser differs.

---

## 1. Where the input streams come from

Everything the monitor needs is observable on the loop's redis instance
(`127.0.0.1:6379`, db 0 — **never** `localhost` on Windows, the IPv6
fallback stalls ~21 s per connection). The monitor is a **passive
observer**: it must not publish MAPLE-K events or write MAPLE-K knowledge.

### Events → boolean input streams

rpclpy publishes each event to the **bare event-key pub/sub channel**
(`anomaly_detected`, *not* `/anomaly_detected` — the `topic:` entries in
`managing_system/config.yaml` are not what is on the wire) with a JSON
`{uid, timestamp}` payload:

| LOLA stream | redis pub/sub channel | emitted by |
|---|---|---|
| `sensor_data_received` | `sensor_data_received` | pendant RPC bridge / sensor tick |
| `observation_recorded` | `observation_recorded` | Monitor |
| `detection_completed` | `detection_completed` | Analysis (every cycle) |
| `anomaly_detected` | `anomaly_detected` | Analysis |
| `plan_generated` | `plan_generated` | Plan |
| `plan_validated` / `plan_rejected` | same names | Legitimate |
| `plan_executed` | `plan_executed` | Execute |

### Knowledge → sampled value streams

Knowledge objects are **pickled Python instances** stored under their class
name as the redis key (classes in `managing_system/messages.py`; rpclpy
stamps `uid`/`timestamp` on write). At every tick, each stream carries the
latest stored value:

| LOLA stream | redis key → field |
|---|---|
| `entropy_len` | `EntropyHistory` → `len(values)` |
| `running_avg` | `RunningAvgEntropy` → `value` |
| `screw_entropy` | `Detections` → `screw_entropy` |
| `brightness` | `FrameRef` → `brightness` |
| `in_planning` | `InPlanning` → `in_planning` |
| `replanning_counter` | `ReplanningCounter` → `count` |
| `active_model` | `ActiveModel` → `model_id` |
| `candidate_model` | `CandidateModel` → `model_id` |
| `legit_is_legit` | `LegitResult` → `is_legit` |
| `legit_candidate_model` | `LegitResult` → `candidate_model_id` |
| `legit_candidate_avg` | `LegitResult` → `candidate_avg_entropy` |
| `legit_current_avg` | `LegitResult` → `current_avg_entropy` |
| `action_model` | `ActionCommand` → `model_id` |

**Sentinels:** `None`/missing numeric values are fed as `-1.0` (ids as
`-1`); booleans default to `false`. The equations rely on this (e.g.
`legit_rule_accepts` requires `legit_candidate_avg >= 0.0`).

### Tick model

* One tick per received bus event (its boolean stream true, all others
  false), **plus** a periodic timer tick at ≥ 1 Hz (`timer_tick` true) so
  timeout properties can fire while the loop is quiet.
* `t` is seconds since monitor start (monotonic float).
* All knowledge streams are (re)sampled at every tick.

**Race-freedom:** every node writes its knowledge *before* publishing the
corresponding event, so sampling knowledge on the tick of an event always
sees the values that event refers to.

---

## 2. Property catalogue

### Before adaptation — is the evidence sound?

| # | Streams | Property |
|---|---|---|
| P1 | `v_capture_timeout`, `v_detection_timeout` | Bounded response `sensor_data_received → observation_recorded → detection_completed`. The RobotBridge blocks the UR pendant on `detection_completed`, and `safe_callback` swallows node exceptions — this makes a silently dead node visible. Assumes at most one outstanding request (the pendant serializes calls). |
| P2 | `v_window_overflow`, `v_entropy_range`, `v_avg_range` | Analysis sanity: window length ≤ configured size, entropies and averages in `[0, 1]`. (Exact mean recomputation needs the raw window contents — a host-side check if the monitor supports it.) |
| P3 | `w_brightness_range` | Frame brightness within a plausible sensor range (outside = camera/lighting fault). Warning, not violation. |
| P4 | `v_anomaly_unsound` | `anomaly_detected` is justified: full window, running average above threshold. |
| P5 | `v_missed_anomaly` | No silent degradation: a full window above threshold with no episode in flight must raise an anomaly within `anomaly_grace`. |

### During adaptation — is the episode protocol honored?

| # | Streams | Property |
|---|---|---|
| P6 | `v_overlapping_anomaly`, `v_unsolicited_plan`, `v_unsolicited_verdict`, `v_unlegitimated_execution` | State-machine conformance `anomaly → plan → (validate\|reject) → execute`; the core property is **no actuation without a validated plan**. |
| P7 | `v_plan_timeout`, `v_legit_timeout`, `v_execute_timeout` | Per-step deadlines (legitimation gets the longest — it re-runs YOLO over the rolling window). |
| P8 | `v_episode_timeout` | Whole-episode bound: every anomaly ends in execute or abort; `InPlanning` never sits wedged true (which would suppress all future anomalies). |
| P9 | `v_candidate_unknown`, `v_candidate_self_swap` | Candidate is a known model id and differs from the active model — a self-swap can never clear the anomaly. |
| P10 | `v_unjustified_accept`, `v_unjustified_reject`, `v_verdict_flag_mismatch` | Legitimate's verdict matches its own rule: accept ⟺ `candidate_avg < current_avg ∨ candidate_avg < threshold`, both directions, and the published `is_legit` flag agrees with the event. |
| P11 | `v_replan_budget`, `v_counter_leak` | ≤ `max_replans` rejections per episode, and `ReplanningCounter` is 0 at episode start (an aborted episode must not leak its count). |

### After adaptation — did it work?

| # | Streams | Property |
|---|---|---|
| P12 | `v_execute_mismatch` | `ActionCommand.model_id = LegitResult.candidate_model_id = ActiveModel.model_id`, and the result really was an accept. |
| P13 | `v_post_reset` | Clean slate after the swap: entropy window empty, `InPlanning` false, counter 0. |
| P14 | `v_ineffective_adaptation` | The adaptation's promise: Legitimate validated on *past* frames; once the window refills with *live* frames, the average must be back under the threshold. |
| P15 | `v_thrashing`, `v_ping_pong` | Bounded swap rate and no `A → B → A` oscillation within the stability window. |

---

## 3. Parameters

The first block of `out` constants in `maple_k.lola` **must mirror
`Adaptation_Config` in `managing_system/config.yaml`** (they are the
deployed adaptation policy):

| LOLA constant | config.yaml key | current value |
|---|---|---|
| `entropy_window_size` | `entropy_window_size` | 10 |
| `entropy_threshold` | `entropy_threshold` | 0.35 |
| `max_replans` | `max_replans` | 3 |
| `min_model_id`/`max_model_id` | model ids in `managed_system` | 1–4 |

The remaining constants (deadlines, grace periods, stability window,
brightness bounds) are RV-specific tuning and can be adjusted freely; the
detection/legitimation deadlines are sized for YOLO on CPU.

---

## 4. Known findings the spec will raise on the current code

Two properties are expected to fire against the code as-is — genuine latent
defects, not spec errors:

* **`v_candidate_self_swap` (P9):** `Plan` proposes the fixed
  `candidate_model_id` regardless of the active model, so if that model is
  already active the candidate equals the incumbent.
* **`v_counter_leak` (P11):** on an aborted episode `Legitimate` never
  resets `ReplanningCounter` (only acceptance does), so the *next* episode
  starts over budget and aborts after a single rejection.

## 5. Modeling notes & assumptions

* **Abort has no event.** A replan-budget abort is only visible as
  `InPlanning` flipping false mid-episode; the `phase` stream falls back to
  idle on that observation instead of timing out spuriously.
* **Level semantics for timeouts.** `v_*_timeout` streams stay true on
  every tick past the deadline until the awaited event arrives — the
  monitor may edge-detect for reporting.
* **Single outstanding request** for P1: the pendant's synchronous XML-RPC
  contract serializes sensor ticks. If the loop is ever driven concurrently
  the capture/detection latency tracking needs a queue, which plain LOLA
  cannot express.
* **P2 mean recomputation** (`RunningAvgEntropy = mean(EntropyHistory)`) is
  left to the monitor host side, since LOLA has no arrays to hold the raw
  window contents; the spec bounds length and range instead.
