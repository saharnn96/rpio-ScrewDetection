# Trustworthiness specification for the MAPLE-K loop

This directory holds the **runtime-verification specification** for this
system: the properties a trustworthiness checker must verify **before,
during and after** each adaptation episode. The checker itself is developed
by the RV team; this is the contract they implement against.

* [`maple_k.dsrv`](maple_k.dsrv) — the specification: 15 properties (P1–P15)
  in the `.dsrv` stream language the RV team's checker runs (`in`/`out`/`aux`
  streams, `Struct<...>`/`List<...>` types, `default()`/`stream[1]` past
  references, `List.fold`). Bus events arrive as one pub/sub struct per
  channel and knowledge as one struct per redis key. Each property is a
  `v_*` (violation) or `w_*` (warning) output stream; `violation` /
  `warning` aggregate them.
* [`maplek_input.json5`](maplek_input.json5) — the input-source config that
  binds those `in` streams to redis: the 8 event channels plus `t` on the
  `redis` source, and the 9 knowledge keys on the `redis-knowledge` source.
* [`tc_dashboard.py`](tc_dashboard.py) and
  [`test_tc_dashboard_redis.py`](test_tc_dashboard_redis.py) — a live view of
  the checker's output streams, and a fake publisher to exercise it (§6).
* [`violation_simulation.py`](violation_simulation.py) — a fault-injecting driver that plays the
  managing system and walks the checker into each violation on purpose (§7).
* This README — how to derive the input streams from the running system, the
  tick model, and the property catalogue in prose.

---

## 1. Where the input streams come from

Everything the checker needs is observable on the loop's redis instance
(`127.0.0.1:6379` — **never** `localhost` on Windows, the IPv6 fallback
stalls ~21 s per connection). Events are pub/sub on db 0; knowledge is read
from keys on the db named in [`maplek_input.json5`](maplek_input.json5)
(currently `database: 2` — check yours with `redis-cli -n 2 KEYS "*"`). The
checker is a **passive observer**: it must not publish MAPLE-K events or
write MAPLE-K knowledge.

### Events → pub/sub input streams

rpclpy publishes each event to the **bare event-key pub/sub channel**
(`anomaly_detected`, *not* `/anomaly_detected` — the `topic:` entries in
`managing_system/config.yaml` are not what is on the wire) with a JSON
`{uid, timestamp}` payload. The spec declares each as
`Struct<uid: Str, timestamp: Str, ...>` and derives a "fired this tick"
boolean by comparing `uid` against the previous tick (the `*_fired` `aux`
streams).

| `.dsrv` stream | redis pub/sub channel | emitted by |
|---|---|---|
| `sensor_data_received` | `sensor_data_received` | pendant RPC bridge / sensor tick |
| `observation_recorded` | `observation_recorded` | Monitor |
| `detection_completed` | `detection_completed` | Analysis (every cycle) |
| `anomaly_detected` | `anomaly_detected` | Analysis |
| `plan_generated` | `plan_generated` | Plan |
| `plan_validated` / `plan_rejected` | same names | Legitimate |
| `plan_executed` | `plan_executed` | Execute |

### Knowledge → sampled struct streams

Knowledge objects are written by rpclpy's `write_knowledge_as_object()` as
JSON under their own redis key (domain classes in
`managing_system/messages.py`; rpclpy stamps `uid`/`timestamp` on write, and
the trailing `...` in each `Struct` accepts those extra fields). Each stream
carries the latest stored value:

| `.dsrv` stream | redis key | fields used |
|---|---|---|
| `frame_ref` | `frame_ref` | `brightness` |
| `entropy_history` | `entropy_history` | `values: List<Float>` |
| `running_avg_entropy` | `running_avg_entropy` | `value` |
| `in_planning` | `in_planning` | `in_planning` |
| `replanning_counter` | `replanning_counter` | `count` |
| `active_model` | `active_model` | `model_id` |
| `candidate_model` | `candidate_model` | `model_id` |
| `legit_result` | `legit_result` | `is_legit`, `candidate_model_id`, `candidate_avg_entropy`, `current_avg_entropy` |
| `action_command` | `action_command` | `model_id` |

The key names on the right are whatever your `KnowledgeManager` actually
writes — they are remapped in `maplek_input.json5`, not hard-coded in the
spec.

**Sentinels:** missing numeric values are read as `-1.0` (ids as `-1`) via
`default()`; booleans default to `false`. The equations rely on this (e.g.
`legit_rule_accepts` requires `candidate_avg_entropy >= 0.0`).

### Tick model

* One tick per received bus event (its stream carrying a fresh `uid`, all
  others unchanged), **plus** a periodic clock tick so timeout properties can
  fire while the loop is quiet.
* `t` is seconds since checker start (monotonic float), published on its own
  redis channel by the checker's heartbeat script — **the timeout properties
  are dead without it**.
* Knowledge streams are (re)sampled whenever their key changes;
  `publish_initial: true` seeds them at startup.

**Race-freedom:** every node writes its knowledge *before* publishing the
corresponding event, so sampling knowledge on the tick of an event always
sees the values that event refers to.

---

## 2. Property catalogue

### Before adaptation — is the evidence sound?

| # | Streams | Property |
|---|---|---|
| P1 | `v_capture_timeout`, `v_detection_timeout` | Bounded response `sensor_data_received → observation_recorded → detection_completed`. The RobotBridge blocks the UR pendant on `detection_completed`, and `safe_callback` swallows node exceptions — this makes a silently dead node visible. Assumes at most one outstanding request (the pendant serializes calls). |
| P2 | `v_window_overflow`, `v_entropy_range`, `v_avg_range`, `v_avg_mismatch` | Analysis sanity: window length ≤ configured size, entropies and averages in `[0, 1]`, and the published `running_avg_entropy` really is the mean of the window (recomputed with `List.fold` over `entropy_history.values`). |
| P3 | `w_brightness_range` | Frame brightness within a plausible sensor range (outside = camera/lighting fault). Warning, not violation. |
| P4 | `v_anomaly_unsound` | `anomaly_detected` is justified: full window, running average above threshold. |
| P5 | `v_missed_anomaly` | No silent degradation: a full window above threshold with no episode in flight must raise an anomaly within `anomaly_grace`. |

### During adaptation — is the episode protocol honored?

| # | Streams | Property |
|---|---|---|
| P6 | `v_overlapping_anomaly`, `v_unsolicited_plan`, `v_unsolicited_verdict`, `v_unlegitimated_execution` | State-machine conformance `anomaly → plan → (validate\|reject) → execute`; the core property is **no actuation without a validated plan**. |
| P7 | `v_plan_timeout`, `v_legit_timeout`, `v_execute_timeout` | Per-step deadlines (legitimation gets the longest — it re-runs YOLO over the rolling window). |
| P8 | `v_episode_timeout` | Whole-episode bound: every anomaly ends in execute or abort; `in_planning` never sits wedged true (which would suppress all future anomalies). |
| P9 | `v_candidate_unknown`, `v_candidate_self_swap` | Candidate is a known model id and differs from the active model — a self-swap can never clear the anomaly. |
| P10 | `v_unjustified_accept`, `v_unjustified_reject`, `v_verdict_flag_mismatch` | Legitimate's verdict matches its own rule: accept ⟺ `candidate_avg < current_avg ∨ candidate_avg < threshold`, both directions, and the published `is_legit` flag agrees with the event. |
| P11 | `v_replan_budget`, `v_counter_leak` | ≤ `max_replans` rejections per episode, and `replanning_counter` is 0 at episode start (an aborted episode must not leak its count). |

### After adaptation — did it work?

| # | Streams | Property |
|---|---|---|
| P12 | `v_execute_mismatch` | `action_command.model_id = legit_result.candidate_model_id = active_model.model_id`, and the result really was an accept. |
| P13 | `v_post_reset` | Clean slate after the swap: entropy window empty, `in_planning` false, counter 0. |
| P14 | `v_ineffective_adaptation` | The adaptation's promise: Legitimate validated on *past* frames; once the window refills with *live* frames, the average must be back under the threshold. |
| P15 | `v_thrashing`, `v_ping_pong` | Bounded swap rate and no `A → B → A` oscillation within the stability window. |

---

## 3. Parameters

The first block of `out` constants in `maple_k.dsrv` **must mirror
`Adaptation_Config` in `managing_system/config.yaml`** (they are the
deployed adaptation policy):

| `.dsrv` constant | config.yaml key | current value |
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
  `in_planning` flipping false mid-episode; the `phase` stream falls back to
  idle on that observation instead of timing out spuriously.
* **Level semantics for timeouts.** `v_*_timeout` streams stay true on
  every tick past the deadline until the awaited event arrives — the
  checker may edge-detect for reporting.
* **Single outstanding request** for P1: the pendant's synchronous XML-RPC
  contract serializes sensor ticks. If the loop is ever driven concurrently
  the capture/detection latency tracking needs a queue, which the stream
  language cannot express.
* **Event identity is by `uid`.** A repeated payload carrying the same `uid`
  is not a new occurrence; the `*_fired` aux streams exist precisely to
  collapse that.

---

## 6. Live property dashboard

[`tc_dashboard.py`](tc_dashboard.py) subscribes to the checker's **output**
streams on redis and renders one rolling timeline bar per property
(green = false, red = true, gray = not yet seen). It expects each `out`
stream of [`maple_k.dsrv`](maple_k.dsrv) to be published on its own bare
channel (`v_capture_timeout`, ...) with a JSON `true` / `false` payload;
prefixed channels (`nodeA/v_capture_timeout`) are matched on their last
segment. On startup it validates that the `.dsrv` file declares every signal
in its `P1..P15` map and exits with the missing names if not.

[`test_tc_dashboard_redis.py`](test_tc_dashboard_redis.py) is a fake
publisher for the same channels, so you can exercise the dashboard without
the MAPLE-K loop or the RV checker running.

Run them from this directory, in two terminals:

```powershell
python -u tc_dashboard.py --dsrv-file maple_k.dsrv --redis-host 127.0.0.1 --redis-port 6379 --port 8090
```

```powershell
python -u test_tc_dashboard_redis.py --redis-host 127.0.0.1 --redis-port 6379 --mode wave --interval 1.0
```

Then open **<http://127.0.0.1:8090>**. `-u` keeps `--verbose` output
unbuffered; without it the dashboard's log appears only in chunks.

### Options worth knowing

`tc_dashboard.py`:

| Flag | Default | Meaning |
|---|---|---|
| `--dsrv-file` | `examples/screw_detection_maple_full.dsrv` | Spec to validate the signal list against — **pass `maple_k.dsrv`**, the default does not exist here |
| `--redis-host` / `--redis-port` / `--redis-db` | `127.0.0.1` / `6379` / `0` | Checker's redis (**never** `localhost` on Windows) |
| `--host` / `--port` | `127.0.0.1` / `8050` | Dash bind address and port |
| `--refresh-ms` | `500` | UI refresh interval |
| `--history-window-sec` / `--history-bins` | `10.0` / `60` | Length and resolution of each timeline bar |
| `--verbose` | off | Log every matched channel/value to stdout |

`test_tc_dashboard_redis.py`:

| Flag | Default | Meaning |
|---|---|---|
| `--mode` | `wave` | `wave` walks a single violation down P1→P15; `random` flips each signal with `--violation-prob`; `all-ok` / `all-bad` hold every signal false / true |
| `--interval` | `1.0` | Seconds between publish rounds |
| `--violation-prob` | `0.05` | Per-signal true probability in `--mode random` |
| `--rounds` | `0` | Stop after N rounds (`0` = until `Ctrl+C`) |
| `--dsrv-file` | none | Optional: cross-check the signal list against a `.dsrv` file before publishing |

It takes its signal list from `tc_dashboard.PROPERTY_SIGNALS`, so the two
cannot drift apart.

**Sanity check without the UI** — publishes two rounds and exits:

```powershell
python test_tc_dashboard_redis.py --dsrv-file maple_k.dsrv --rounds 2 --interval 0.3
```

Requires `dash` and `redis` (both in [`requirements.txt`](../requirements.txt))
and a redis server on `127.0.0.1:6379`.


---

## 7. Fault-injection driver

[`simulation.py`](simulation.py) is the *input* side of the checker: it
publishes the 8 event channels, the `t` clock and the 9 knowledge keys itself,
following the protocol faithfully except in one deliberate place per scenario.
It exists because most of these properties cannot be provoked from the real
loop without patching production code — you cannot easily ask `Legitimate` to
contradict its own verdict, or wedge `Analysis` on demand.

```powershell
python simulation.py --list
python simulation.py --scenario counter-leak
python simulation.py --scenario all
```

25 scenarios, one per violation stream (`--list` prints the catalogue with the
property each one targets). Run the checker and
[`tc_dashboard.py`](tc_dashboard.py) alongside it and the matching bar turns
red.

**Virtual clock.** `t` comes from this process, so time is ours: a wait jumps
straight to its target in one tick, turning the 120 s `episode_deadline` into a
single tick rather than a two-minute sit. The spec's deadlines are untouched —
this works because every `v_*_timeout` is level-triggered, so landing one tick
past a deadline is enough to observe it.

| Flag | Default | Meaning |
|---|---|---|
| `--scenario` | `nominal` | scenario name, or `all` |
| `--tick-step` | `1.0` | virtual seconds a plain tick advances |
| `--tick-interval` | `0.35` | real seconds between ticks — see pacing below |
| `--settle-ticks` | `2` | ticks between writing knowledge and publishing the event it explains |
| `--start-t` | resume | force the virtual clock instead of resuming from `sim:last_t` |
| `--enable-keyspace-events` | off | set `notify-keyspace-events=KA` if it is off |
| `--event-db` / `--knowledge-db` | `0` / `2` | must match `maplek_input.json5` |
| `--repeat` | `1` | run N times (`0` = until `Ctrl+C`) |
| `--dry-run` | off | print the timeline, touch no redis |

Each scenario ends with a recovery step that walks the checker back to phase 0
without inventing a second finding, and `--scenario all` inserts a
`stability_window` cooldown between scenarios so P15 does not bleed across
them.

### Getting a violation to actually appear

Four things must be true, and each one fails **silently** — the checker keeps
publishing `false` rather than erroring.

1. **`notify-keyspace-events` must be on.** The checker learns about knowledge
   writes by subscribing to `__keyspace@2__:<key>`. Redis emits nothing there
   by default, so the checker freezes every knowledge stream at whatever
   `publish_initial` handed it and 13 of the 15 properties can never fire.
   Fix once with `redis-cli CONFIG SET notify-keyspace-events KA` (add
   `notify-keyspace-events KA` to `redis.conf` to survive a restart).
   `simulation.py` checks this at startup and warns.
2. **`t` must be monotonic for the checker's whole lifetime.** Every deadline
   is a `t - stored_timestamp` subtraction against values the checker is still
   holding. Restart the clock at 0 and those differences go negative forever.
   The clock is checkpointed in `sim:last_t`; use `--start-t 0` only when
   restarting the checker too.
3. **Knowledge needs a moment to land before the event that explains it.**
   Knowledge arrives via keyspace notifications and events via pub/sub — two
   independent streams. Publishing an anomaly in the same instant as the
   counter write that makes it a violation lets the checker evaluate the event
   against stale knowledge. `--settle-ticks 2` covers it; `0` reproduces the
   race.
4. **Do not outrun the checker.** It evaluates one tick per published `t`, at
   roughly 2–3 ticks/s in a debug build. Publish faster and you build a
   backlog that puts it minutes behind while it still looks alive. Waits jump
   to their target in a single tick rather than stepping, which keeps a whole
   scenario to a few dozen ticks.

### Things worth knowing

* **Scenarios are isolated on purpose.** Each one was checked against a
  reference evaluation of `maple_k.dsrv`: it raises its own property and no
  other. `nominal` raises nothing at all — use it to confirm the checker is
  quiet on a healthy loop before trusting a red bar.
* **Two properties cannot be isolated, and that is the spec talking.**
  `v_ping_pong` always co-fires with `v_thrashing` — both compare the
  third-most-recent swap against the same `stability_window`, so an `A → B → A`
  oscillation is necessarily three swaps inside it. That is a real observation
  about the specification, worth raising with the RV team.
* **Tick ordering is an assumption.** Each step writes knowledge, publishes its
  event, then publishes the new `t`. If the checker instead ticks on the clock
  and reads what has arrived *since*, use `--clock-first`.
* **db 2 holds unrelated leftovers** from earlier experiments on this machine
  (`reading:*`, pickled rpclpy objects). Harmless — the checker only reads the
  9 keys named in `maplek_input.json5`.
* **One redis, not two.** WSL runs with mirrored networking here, so Windows
  `127.0.0.1:6379` and WSL `127.0.0.1:6379` are the same server. The checker
  runs under WSL; the simulator and dashboard run on Windows; they meet.

### Findings from running this against the real checker

All 25 scenarios were confirmed to raise their target property on the live
checker (`target/debug/trustworthiness_checker maple_k.dsrv --input-config
... --redis-output`). Three things surfaced that are worth the RV team's
attention:

* **`v_avg_mismatch` (P2) is racy by construction.** `entropy_history` and
  `running_avg_entropy` are two separate redis keys with two separate
  notifications, so between them the checker always sees a new window against
  the old average and flags a mismatch. It fired as collateral in nearly every
  scenario. The real Analysis node writes them the same way, so this will
  produce false positives in production. It needs either a single combined
  knowledge object or hysteresis (require the mismatch on N consecutive ticks).
* **`v_ping_pong` cannot fire without `v_thrashing`.** Both compare the
  third-most-recent swap against the same `stability_window`, so P15's two
  streams are not independent.
* **The checker hangs silently after a few thousand ticks.** It stops
  publishing entirely while the process stays alive, with nothing on stdout and
  no CPU burn; it then ignores all further input. Reproduced on two separate
  instances. Restarting it is the only recovery found. Long runs need a
  restart between batches.
