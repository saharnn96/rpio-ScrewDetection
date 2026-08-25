#!/usr/bin/env python3
"""Fault-injecting driver for the MAPLE-K trustworthiness checker.

Plays the role of the whole managing system: publishes the 8 MAPLE-K event
channels, the `t` clock and the 9 knowledge keys that `maple_k.dsrv` reads,
following the real protocol -- except that each scenario bends it in exactly
one place so a named property fires.

This is the *input* side of the checker. `test_tc_dashboard_redis.py` fakes
the *output* side; this one makes a real checker produce those outputs.

    python violaition_simulation.py --list
    python violaition_simulation.py --scenario counter-leak
    python violaition_simulation.py --scenario all

Nothing here imports the managing system -- the point is to reach states the
real loop cannot easily be coaxed into (a wedged node, a leaked counter, a
model ping-pong) without patching production code.

VIRTUAL CLOCK
-------------
The checker's `t` comes from this process, so time is ours to speed up:
`--speed 20` advances 20 virtual seconds per real second, which turns the
120 s episode deadline into a 6 s wait. Deadlines in the spec are unchanged;
only the wall-clock cost of reaching them is.

`t` MUST be monotonic for as long as the checker lives: it stores timestamps
(`sensor_t`, `episode_start_t`, `swap1_t`, ...) and every deadline is a
`t - stored` subtraction. Restarting this script from t=0 puts those stored
values in the future, every difference goes negative, and no timeout can ever
fire again. So the clock is checkpointed in redis under `sim:last_t` and each
run continues from there. `--start-t 0` resets it -- only correct if you
restart the checker too.
"""

from __future__ import annotations

import argparse
import json
import re
import threading
import time
import uuid
from collections import Counter
from datetime import datetime, timezone
from typing import Callable

import redis

# --- constants mirrored from maple_k.dsrv (see README section 3) -----------
ENTROPY_WINDOW_SIZE = 10
ENTROPY_THRESHOLD = 0.35
MAX_REPLANS = 3
MIN_MODEL_ID = 1
MAX_MODEL_ID = 4
ANOMALY_GRACE = 10.0
STABILITY_WINDOW = 600.0
BRIGHTNESS_MIN = 5.0
BRIGHTNESS_MAX = 250.0

# --- how long each violation stream stays true -----------------------------
# EDGE streams are written `<event>_fired && <bad condition>`, so they are true
# for exactly the one tick the event arrives and false again immediately after.
# LEVEL streams are conditions over stored state and stay true every tick until
# something clears them.
#
# This is the single most important thing to know when a scenario "works" and
# the dashboard still looks green: 18 of the 29 streams are one-tick pulses, and
# a UI that renders the *current* value at 2 Hz will simply never sample them.
# Only the LEVEL streams can be held long enough to watch (see Sim.hold).
EDGE, LEVEL = "edge", "level"

STREAM_KIND: dict[str, str] = {
    # P1
    "v_capture_timeout": LEVEL,
    "v_detection_timeout": LEVEL,
    # P2 -- pure knowledge predicates, true while the bad window is published
    "v_window_overflow": LEVEL,
    "v_entropy_range": LEVEL,
    "v_avg_range": LEVEL,
    "v_avg_mismatch": LEVEL,
    # P3
    "w_brightness_range": EDGE,
    # P4
    "v_anomaly_unsound": EDGE,
    # P5
    "v_missed_anomaly": LEVEL,
    # P6
    "v_overlapping_anomaly": EDGE,
    "v_unsolicited_plan": EDGE,
    "v_unsolicited_verdict": EDGE,
    "v_unlegitimated_execution": EDGE,
    # P7
    "v_plan_timeout": LEVEL,
    "v_legit_timeout": LEVEL,
    "v_execute_timeout": LEVEL,
    # P8
    "v_episode_timeout": LEVEL,
    # P9
    "v_candidate_unknown": EDGE,
    "v_candidate_self_swap": EDGE,
    # P10
    "v_unjustified_accept": EDGE,
    "v_unjustified_reject": EDGE,
    "v_verdict_flag_mismatch": EDGE,
    # P11
    "v_replan_budget": EDGE,
    "v_counter_leak": EDGE,
    # P12
    "v_execute_mismatch": EDGE,
    # P13
    "v_post_reset": EDGE,
    # P14 -- guarded by assessing_effect[1], which flips false on the same tick
    "v_ineffective_adaptation": EDGE,
    # P15
    "v_thrashing": EDGE,
    "v_ping_pong": EDGE,
}

EVENT_CHANNELS = (
    "sensor_data_received",
    "observation_recorded",
    "detection_completed",
    "anomaly_detected",
    "plan_generated",
    "plan_validated",
    "plan_rejected",
    "plan_executed",
)

# Knowledge key -> the value the spec expects when nothing has happened yet.
BASELINE_KNOWLEDGE: dict[str, dict] = {
    "frame_ref": {"brightness": 128.0},
    "entropy_history": {"values": []},
    "running_avg_entropy": {"value": 0.0},
    "in_planning": {"in_planning": False},
    "replanning_counter": {"count": 0},
    "active_model": {"model_id": 1},
    "candidate_model": {"model_id": 2},
    "legit_result": {
        "is_legit": False,
        "candidate_model_id": -1,
        "candidate_avg_entropy": -1.0,
        "current_avg_entropy": -1.0,
    },
    "action_command": {"model_id": -1},
}

# Phase encoding from the spec: 0 idle, 1 anomaly, 2 planned, 3 validated.
IDLE, ANOMALY, PLANNED, VALIDATED = 0, 1, 2, 3


def window_mean(values: list[float]) -> float:
    """Mean the way the spec folds it, so `v_avg_mismatch` stays quiet."""
    if not values:
        return 0.0
    total = 0.0
    for value in values:
        total += value
    length = 0.0
    for _ in values:
        length += 1.0
    return total / length


class Observer:
    """Watches the checker's own output channels so a scenario can self-check.

    The dashboard shows each stream's *current* value, so an EDGE violation --
    true for one tick out of a few hundred -- is easy to miss completely. This
    subscribes to the same channels and counts rising edges, which is what lets
    a scenario report "v_counter_leak fired" instead of leaving you to catch a
    red flash. Redis pub/sub is not database-scoped, so the db here is
    irrelevant: it hears whatever the checker publishes with --redis-output.
    """

    def __init__(self, client: redis.Redis, streams: list[str]) -> None:
        self.lock = threading.Lock()
        self.rising: Counter = Counter()
        self.messages = 0
        self._last: dict[str, bool] = {}
        self._pubsub = client.pubsub(ignore_subscribe_messages=True)
        self._pubsub.subscribe(*streams)
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    @staticmethod
    def _decode(raw) -> bool | None:
        text = raw.decode("utf-8", "replace") if isinstance(raw, bytes) else str(raw)
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            lowered = text.strip().lower()
            return True if lowered == "true" else False if lowered == "false" else None
        return parsed if isinstance(parsed, bool) else None

    def _run(self) -> None:
        try:
            for message in self._pubsub.listen():
                if message.get("type") != "message":
                    continue
                channel = message["channel"]
                name = channel.decode() if isinstance(channel, bytes) else str(channel)
                value = self._decode(message.get("data"))
                if value is None:
                    continue
                with self.lock:
                    self.messages += 1
                    # Count the transition, not the sample: the checker
                    # republishes every stream on every tick.
                    if value and not self._last.get(name, False):
                        self.rising[name] += 1
                    self._last[name] = value
        except Exception:  # the connection dies when the process exits
            pass

    def snapshot(self) -> Counter:
        with self.lock:
            return Counter(self.rising)


class Sim:
    """Drives redis on a virtual clock, tracking the phase the spec will infer."""

    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.vt = 0.0
        self.phase = IDLE
        self.clock_key = "sim:last_t"
        self.knowledge_dirty = False
        self.knowledge = {key: dict(value) for key, value in BASELINE_KNOWLEDGE.items()}
        self.log: list[str] = []
        self.ticks = 0

        if args.dry_run:
            self.events = self.kb = None
        else:
            self.events = redis.Redis(
                host=args.redis_host, port=args.redis_port, db=args.event_db
            )
            self.kb = redis.Redis(
                host=args.redis_host, port=args.redis_port, db=args.knowledge_db
            )
            self.events.ping()
            self.kb.ping()
            self._check_keyspace_events()

        self.observer = (
            None
            if self.events is None or args.no_observe
            else Observer(self.events, sorted(STREAM_KIND))
        )
        self.vt = self._resume_clock()

    def _check_keyspace_events(self) -> None:
        """Knowledge only reaches the checker via keyspace notifications.

        The checker learns about knowledge writes by subscribing to
        `__keyspace@2__:<key>`. Redis emits nothing on those channels unless
        `notify-keyspace-events` is configured, and its default is off -- in
        which case the checker silently freezes every knowledge stream at
        whatever `publish_initial` gave it and 13 of the 15 properties can
        never fire. Nothing errors; the bars just stay green.
        """
        flags = self.events.config_get("notify-keyspace-events").get(
            "notify-keyspace-events", ""
        )
        ok = "K" in flags and ("A" in flags or "$" in flags)
        if ok:
            return
        if self.args.enable_keyspace_events:
            self.events.config_set("notify-keyspace-events", "KA")
            print("[sim] enabled redis notify-keyspace-events=KA", flush=True)
            return
        print(
            f"[sim] WARNING: redis notify-keyspace-events={flags!r} -- the checker "
            "cannot see knowledge writes, so only event/timing properties can "
            "fire.\n[sim]          re-run with --enable-keyspace-events, or "
            "redis-cli CONFIG SET notify-keyspace-events KA",
            flush=True,
        )

    def _resume_clock(self) -> float:
        """Continue the virtual clock where the last run left off.

        The checker outlives this process, so `t` has to keep rising across
        runs; the 5 s gap keeps successive runs from sharing an instant.
        """
        if self.args.start_t is not None:
            return self.args.start_t
        if self.events is None:
            return 0.0
        stored = self.events.get(self.clock_key)
        return float(stored) + 5.0 if stored else 0.0

    # -- plumbing ----------------------------------------------------------

    def _stamp(self) -> dict:
        return {
            "uid": uuid.uuid4().hex,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    def say(self, text: str) -> None:
        line = f"[{self.vt:8.2f}s] {text}"
        self.log.append(line)
        if not self.args.quiet:
            print(line, flush=True)

    def _publish_clock(self) -> None:
        if self.events is not None:
            self.events.publish("t", json.dumps(round(self.vt, 3)))

    def write(self, key: str, **fields) -> None:
        """SET one knowledge key. Always called *before* the event it explains."""
        self.knowledge[key].update(fields)
        self.knowledge_dirty = True
        if self.kb is not None:
            payload = dict(self.knowledge[key], **self._stamp())
            self.kb.set(key, json.dumps(payload))

    def set_window(self, values: list[float], avg: float | None = None) -> None:
        """Set the entropy window and its average together.

        `avg=None` keeps them consistent; pass a number to inject a P2 mismatch.
        """
        self.write("entropy_history", values=list(values))
        self.write(
            "running_avg_entropy",
            value=window_mean(values) if avg is None else avg,
        )

    def emit(self, channel: str, **extra) -> None:
        """Publish one bus event and mirror the spec's phase transition."""
        assert channel in EVENT_CHANNELS, channel

        # The checker reads knowledge over keyspace notifications and events
        # over pub/sub -- two independent streams. Publishing an event in the
        # same instant as the SET that justifies it lets the checker see the
        # event against stale knowledge, so let the writes settle first.
        if self.knowledge_dirty:
            for _ in range(self.args.settle_ticks):
                self.tick()
            self.knowledge_dirty = False

        def send() -> None:
            if self.events is not None:
                self.events.publish(channel, json.dumps(dict(self._stamp(), **extra)))

        if self.args.clock_first:
            self.tick()
            send()
        else:
            send()
            self.tick()

        if channel == "anomaly_detected":
            self.phase = ANOMALY
        elif channel == "plan_generated":
            self.phase = PLANNED
        elif channel == "plan_rejected":
            self.phase = ANOMALY
        elif channel == "plan_validated":
            self.phase = VALIDATED
        elif channel == "plan_executed":
            self.phase = IDLE

        self.say(f"event  {channel}")

    def tick(self, step: float | None = None) -> None:
        """One clock tick: advance virtual time and republish `t`.

        Every published `t` costs the checker one full evaluation, so ticks are
        spent deliberately rather than streamed.
        """
        self.vt += self.args.tick_step if step is None else step
        self.ticks += 1
        self._publish_clock()
        if self.events is not None:
            self.events.set(self.clock_key, repr(self.vt))
        if not self.args.dry_run:
            time.sleep(self.args.tick_interval)

    def advance(self, virtual_seconds: float, note: str | None = None) -> None:
        """Jump N *virtual* seconds in a single tick.

        Timeouts in the spec are level-triggered (`v_*_timeout` stays true every
        tick past the deadline until the awaited event arrives), so landing one
        tick beyond the deadline is enough -- no need to step through it.
        """
        if note:
            self.say(f"wait   {virtual_seconds:.0f}s virtual - {note}")
        self.tick(step=virtual_seconds)

    def hold(self, note: str, ticks: int | None = None) -> None:
        """Sit still so a LEVEL violation stays true long enough to be seen.

        Nothing changes here: only `t` advances. Every `v_*_timeout` and every
        knowledge predicate re-evaluates to true for as long as we do not clear
        the state, which turns a one-tick blip into a bar you can read. Useless
        for EDGE streams -- those are one tick by construction, no matter what
        this driver does.
        """
        count = self.args.hold_ticks if ticks is None else ticks
        if count <= 0:
            return
        self.say(f"hold   {count} ticks - {note}")
        for _ in range(count):
            self.tick()

    def other_model(self) -> int:
        """A valid model id that is not the active one.

        Scenarios used to hard-code candidate 3 while an earlier scenario had
        already left `active_model` at 3, so `--scenario all` raised
        `v_candidate_self_swap` as collateral in eight different places.
        """
        active = self.knowledge["active_model"]["model_id"]
        for candidate in range(MIN_MODEL_ID, MAX_MODEL_ID + 1):
            if candidate != active:
                return candidate
        return MAX_MODEL_ID

    # -- reusable protocol fragments ---------------------------------------

    def healthy_round(self, entropy: float = 0.20) -> None:
        """One clean sense-analyse round: sensor -> observation -> detection."""
        self.emit("sensor_data_received", detection_type="screw")
        self.emit("observation_recorded")
        window = (self.knowledge["entropy_history"]["values"] + [entropy])[
            -ENTROPY_WINDOW_SIZE:
        ]
        self.set_window(window)
        self.emit("detection_completed")

    def fill_window(self, entropy: float, count: int = ENTROPY_WINDOW_SIZE) -> None:
        """Refill the entropy window one sample per tick, as Analysis would."""
        window: list[float] = []
        for _ in range(count):
            window.append(entropy)
            self.set_window(window)
            self.tick()

    def recover(self) -> None:
        """Return the checker to phase 0 without inventing a new violation."""
        if self.phase in (ANOMALY, PLANNED):
            # The silent-abort path: InPlanning drops, the spec falls back to idle.
            self.write("in_planning", in_planning=False)
            self.phase = IDLE
            self.say("recover: in_planning=False -> phase idle")
            self.tick()
        elif self.phase == VALIDATED:
            # Close the episode consistently so the recovery itself is not a finding.
            candidate = self.knowledge["candidate_model"]["model_id"]
            self.write(
                "legit_result",
                is_legit=True,
                candidate_model_id=candidate,
                candidate_avg_entropy=0.20,
                current_avg_entropy=0.55,
            )
            self.finish_episode(candidate)
            # P14 watches from a swap until the window refills. Refill it with a
            # healthy value here, or assessing_effect stays armed and the *next*
            # scenario's degraded window raises v_ineffective_adaptation for free.
            self.fill_window(0.20)

        self.write("replanning_counter", count=0)
        self.write("in_planning", in_planning=False)
        # Back to the baseline model: `active_model` survives `recover` otherwise,
        # and a later scenario picking the same candidate raises P9 for free.
        self.write("active_model", model_id=BASELINE_KNOWLEDGE["active_model"]["model_id"])
        self.set_window([])
        self.tick()

    def open_episode(self, *, counter: int = 0, sound: bool = True) -> None:
        """Reach a justified `anomaly_detected` (full window, average over threshold)."""
        if sound:
            self.fill_window(0.55)
        self.write("replanning_counter", count=counter)
        self.write("in_planning", in_planning=True)
        self.emit("anomaly_detected")

    def finish_episode(self, new_model: int, *, action_model: int | None = None,
                       counter: int = 0, window: list[float] | None = None) -> None:
        """Execute the swap. Defaults satisfy P12/P13; override to break them."""
        self.set_window(window if window is not None else [])
        self.write("in_planning", in_planning=False)
        self.write("replanning_counter", count=counter)
        self.write("active_model", model_id=new_model)
        self.write(
            "action_command",
            model_id=new_model if action_model is None else action_model,
        )
        self.emit("plan_executed")

    def clean_episode(self, new_model: int) -> None:
        """A whole textbook adaptation: anomaly -> plan -> validate -> execute."""
        self.open_episode()
        self.advance(3.0)
        self.write("candidate_model", model_id=new_model)
        self.emit("plan_generated")
        self.advance(4.0)
        self.write(
            "legit_result",
            is_legit=True,
            candidate_model_id=new_model,
            candidate_avg_entropy=0.20,
            current_avg_entropy=0.55,
        )
        self.emit("plan_validated")
        self.advance(3.0)
        self.finish_episode(new_model)
        # P14 watches until the window refills; keep it under threshold.
        self.fill_window(0.20)

    def reject_cycle(self, *, plan_gap: float = 2.0, verdict_gap: float = 2.0,
                     counter: int = 1) -> None:
        """plan -> reject, with a legit_result whose own rule really does reject.

        `plan_gap` is spent in phase 1 (10 s deadline) and `verdict_gap` in
        phase 2 (60 s) -- keep each under its own deadline to isolate P8/P11.
        """
        candidate = self.other_model()
        self.advance(plan_gap)
        self.write("candidate_model", model_id=candidate)
        self.emit("plan_generated")
        self.advance(verdict_gap)
        self.write(
            "legit_result",
            is_legit=False,
            candidate_model_id=candidate,
            candidate_avg_entropy=0.60,
            current_avg_entropy=0.50,
        )
        self.write("replanning_counter", count=counter)
        self.emit("plan_rejected")


# ---------------------------------------------------------------------------
# Scenarios. Each is (property, one-line description, function).
# ---------------------------------------------------------------------------


def sc_nominal(sim: Sim) -> None:
    for _ in range(3):
        sim.healthy_round()
    sim.clean_episode(new_model=2)
    for _ in range(3):
        sim.healthy_round()


def sc_capture_timeout(sim: Sim) -> None:
    sim.emit("sensor_data_received", detection_type="screw")
    sim.advance(8.0, "Monitor never answers (capture_deadline is 5s)")
    sim.hold("v_capture_timeout holds until observation_recorded arrives")
    sim.emit("observation_recorded")  # late, clears the flag
    sim.emit("detection_completed")


def sc_detection_timeout(sim: Sim) -> None:
    sim.emit("sensor_data_received", detection_type="screw")
    sim.emit("observation_recorded")
    sim.advance(25.0, "Analysis is wedged (detection_deadline is 20s)")
    sim.hold("v_detection_timeout holds until detection_completed arrives")
    sim.emit("detection_completed")


def sc_window_overflow(sim: Sim) -> None:
    sim.set_window([0.2] * (ENTROPY_WINDOW_SIZE + 2))
    sim.advance(3.0, "window holds 12 samples, configured size is 10")
    sim.hold("v_window_overflow holds while the oversized window is published")
    sim.set_window([])


def sc_entropy_range(sim: Sim) -> None:
    # 1.7 is outside [0,1]; the other samples keep the mean low so nothing else fires.
    sim.set_window([0.1] * 9 + [1.7])
    sim.advance(3.0, "one sample outside [0,1]")
    sim.hold("v_entropy_range holds while the bad sample is in the window")
    sim.set_window([])


def sc_avg_range(sim: Sim) -> None:
    # Empty window: len 0 disables the mismatch check, isolating v_avg_range.
    sim.set_window([], avg=1.5)
    sim.advance(3.0, "running average 1.5 with an empty window")
    sim.hold("v_avg_range holds while the average stays above 1.0")
    sim.set_window([])


def sc_avg_mismatch(sim: Sim) -> None:
    sim.write("in_planning", in_planning=True)  # suppress P5 while we sit here
    sim.set_window([0.2] * ENTROPY_WINDOW_SIZE, avg=0.9)
    sim.advance(3.0, "published average 0.9, true mean 0.2")
    sim.hold("v_avg_mismatch holds while window and average disagree")
    sim.set_window([])
    sim.write("in_planning", in_planning=False)


def sc_brightness(sim: Sim) -> None:
    sim.emit("sensor_data_received", detection_type="screw")
    sim.write("frame_ref", brightness=BRIGHTNESS_MIN - 3.0)
    sim.emit("observation_recorded")  # warning is sampled on this tick
    sim.write("frame_ref", brightness=128.0)
    sim.emit("detection_completed")


def sc_anomaly_unsound(sim: Sim) -> None:
    sim.set_window([0.2] * 4)  # only 4 samples, and the mean is under threshold
    sim.write("in_planning", in_planning=True)
    sim.emit("anomaly_detected")
    sim.advance(2.0, "anomaly raised on a quarter-full, healthy window")


def sc_missed_anomaly(sim: Sim) -> None:
    sim.fill_window(0.55)  # degraded and nobody is adapting
    sim.advance(ANOMALY_GRACE + 4.0, "degraded window, no anomaly raised")
    sim.hold("v_missed_anomaly holds until the window is cleared")
    sim.set_window([])


def sc_unlegitimated_execution(sim: Sim) -> None:
    # Everything else about the execution is consistent -- only the protocol is broken.
    candidate = sim.other_model()
    sim.write(
        "legit_result",
        is_legit=True,
        candidate_model_id=candidate,
        candidate_avg_entropy=0.20,
        current_avg_entropy=0.55,
    )
    sim.finish_episode(candidate)  # straight from idle: no anomaly, plan or verdict
    sim.fill_window(0.20)


def sc_unsolicited_plan(sim: Sim) -> None:
    sim.write("candidate_model", model_id=sim.other_model())
    sim.emit("plan_generated")  # phase is idle, nobody asked for a plan
    sim.advance(2.0)


def sc_plan_timeout(sim: Sim) -> None:
    sim.open_episode()
    sim.advance(13.0, "Plan never responds (plan_deadline is 10s)")
    sim.hold("v_plan_timeout holds while phase 1 keeps waiting")


def sc_episode_timeout(sim: Sim) -> None:
    # Keep the per-step clocks fresh by cycling plan/reject, so only the
    # whole-episode bound is exceeded. Two rejections stay inside the budget.
    candidate = sim.other_model()
    sim.open_episode()
    sim.reject_cycle(plan_gap=3.0, verdict_gap=50.0, counter=1)
    sim.reject_cycle(plan_gap=3.0, verdict_gap=50.0, counter=2)
    sim.advance(3.0)
    sim.write("candidate_model", model_id=candidate)
    sim.emit("plan_generated")
    sim.advance(20.0, "episode past 120s while each step is still inside its deadline")
    sim.hold("v_episode_timeout holds until the episode closes")


def sc_candidate_self_swap(sim: Sim) -> None:
    sim.open_episode()
    sim.advance(2.0)
    active = sim.knowledge["active_model"]["model_id"]
    sim.write("candidate_model", model_id=active)  # swap the model for itself
    sim.emit("plan_generated")
    sim.advance(2.0)


def sc_candidate_unknown(sim: Sim) -> None:
    sim.open_episode()
    sim.advance(2.0)
    sim.write("candidate_model", model_id=MAX_MODEL_ID + 5)
    sim.emit("plan_generated")
    sim.advance(2.0)


def sc_unjustified_accept(sim: Sim) -> None:
    candidate = sim.other_model()
    sim.open_episode()
    sim.advance(2.0)
    sim.write("candidate_model", model_id=candidate)
    sim.emit("plan_generated")
    sim.advance(3.0)
    # Candidate is worse than the incumbent and over threshold: the rule says reject.
    sim.write(
        "legit_result",
        is_legit=True,
        candidate_model_id=candidate,
        candidate_avg_entropy=0.60,
        current_avg_entropy=0.50,
    )
    sim.emit("plan_validated")
    sim.advance(3.0)
    sim.finish_episode(candidate)
    sim.fill_window(0.20)


def sc_verdict_flag_mismatch(sim: Sim) -> None:
    candidate = sim.other_model()
    sim.open_episode()
    sim.advance(2.0)
    sim.write("candidate_model", model_id=candidate)
    sim.emit("plan_generated")
    sim.advance(3.0)
    # The rule accepts and the event says validated, but the published flag says no.
    sim.write(
        "legit_result",
        is_legit=False,
        candidate_model_id=candidate,
        candidate_avg_entropy=0.20,
        current_avg_entropy=0.55,
    )
    sim.emit("plan_validated")
    sim.advance(3.0)


def sc_replan_budget(sim: Sim) -> None:
    sim.open_episode()
    for attempt in range(1, MAX_REPLANS + 2):  # the 4th rejection breaks the budget
        sim.reject_cycle(plan_gap=2.0, verdict_gap=2.0, counter=attempt)


def sc_counter_leak(sim: Sim) -> None:
    # The known defect: an aborted episode left the counter at 2.
    sim.fill_window(0.55)
    sim.write("replanning_counter", count=2)
    sim.write("in_planning", in_planning=True)
    sim.emit("anomaly_detected")
    sim.advance(2.0)


def sc_execute_mismatch(sim: Sim) -> None:
    candidate = sim.other_model()
    sim.open_episode()
    sim.advance(2.0)
    sim.write("candidate_model", model_id=candidate)
    sim.emit("plan_generated")
    sim.advance(3.0)
    sim.write(
        "legit_result",
        is_legit=True,
        candidate_model_id=candidate,
        candidate_avg_entropy=0.20,
        current_avg_entropy=0.55,
    )
    sim.emit("plan_validated")
    sim.advance(3.0)
    sim.finish_episode(candidate, action_model=MAX_MODEL_ID)  # actuated a different model
    sim.fill_window(0.20)


def sc_post_reset(sim: Sim) -> None:
    candidate = sim.other_model()
    sim.open_episode()
    sim.advance(2.0)
    sim.write("candidate_model", model_id=candidate)
    sim.emit("plan_generated")
    sim.advance(3.0)
    sim.write(
        "legit_result",
        is_legit=True,
        candidate_model_id=candidate,
        candidate_avg_entropy=0.20,
        current_avg_entropy=0.55,
    )
    sim.emit("plan_validated")
    sim.advance(3.0)
    # Swap done, but the window was never cleared and the counter still holds 1.
    sim.finish_episode(candidate, counter=1, window=[0.2, 0.2, 0.2])
    sim.write("replanning_counter", count=0)
    sim.fill_window(0.20)


def sc_ineffective_adaptation(sim: Sim) -> None:
    candidate = sim.other_model()
    sim.open_episode()
    sim.advance(2.0)
    sim.write("candidate_model", model_id=candidate)
    sim.emit("plan_generated")
    sim.advance(3.0)
    sim.write(
        "legit_result",
        is_legit=True,
        candidate_model_id=candidate,
        candidate_avg_entropy=0.20,
        current_avg_entropy=0.55,
    )
    sim.emit("plan_validated")
    sim.advance(3.0)
    sim.finish_episode(candidate)
    # The new model is no better: the refilled window is still over threshold.
    sim.fill_window(0.55)
    sim.set_window([])


def sc_overlapping_anomaly(sim: Sim) -> None:
    # A second anomaly while the first episode is still open (phase 1, not 0).
    sim.open_episode()
    sim.advance(2.0)
    sim.emit("anomaly_detected")
    sim.advance(2.0)


def sc_unsolicited_verdict(sim: Sim) -> None:
    # A verdict with no plan to judge. The verdict itself is internally sound,
    # so only the protocol stream fires.
    candidate = sim.other_model()
    sim.write(
        "legit_result",
        is_legit=True,
        candidate_model_id=candidate,
        candidate_avg_entropy=0.20,
        current_avg_entropy=0.55,
    )
    sim.emit("plan_validated")  # phase is idle, nobody asked for a verdict
    sim.advance(2.0)


def sc_legit_timeout(sim: Sim) -> None:
    sim.open_episode()
    sim.advance(2.0)
    sim.write("candidate_model", model_id=sim.other_model())
    sim.emit("plan_generated")
    sim.advance(65.0, "Legitimate never answers (legit_deadline is 60s)")
    sim.hold("v_legit_timeout holds while phase 2 keeps waiting")


def sc_execute_timeout(sim: Sim) -> None:
    candidate = sim.other_model()
    sim.open_episode()
    sim.advance(2.0)
    sim.write("candidate_model", model_id=candidate)
    sim.emit("plan_generated")
    sim.advance(3.0)
    sim.write(
        "legit_result",
        is_legit=True,
        candidate_model_id=candidate,
        candidate_avg_entropy=0.20,
        current_avg_entropy=0.55,
    )
    sim.emit("plan_validated")
    sim.advance(13.0, "Execute never actuates (execute_deadline is 10s)")
    sim.hold("v_execute_timeout holds while phase 3 keeps waiting")


def sc_unjustified_reject(sim: Sim) -> None:
    candidate = sim.other_model()
    sim.open_episode()
    sim.advance(2.0)
    sim.write("candidate_model", model_id=candidate)
    sim.emit("plan_generated")
    sim.advance(3.0)
    # The rule plainly accepts this candidate -- rejecting it is the violation.
    # is_legit stays false so the verdict agrees with the event (no P10 flag
    # mismatch on top).
    sim.write(
        "legit_result",
        is_legit=False,
        candidate_model_id=candidate,
        candidate_avg_entropy=0.20,
        current_avg_entropy=0.55,
    )
    sim.write("replanning_counter", count=1)
    sim.emit("plan_rejected")
    sim.advance(2.0)


def sc_thrashing(sim: Sim) -> None:
    for model in (2, 3, MAX_MODEL_ID):  # three swaps well inside the 600s window
        sim.clean_episode(model)


def sc_ping_pong(sim: Sim) -> None:
    # v_ping_pong always co-fires with v_thrashing: both compare the third-oldest
    # swap against the same stability window, so the oscillation cannot be
    # expressed without three swaps inside it.
    for model in (2, 3, 2):  # ... and back to where we started
        sim.clean_episode(model)


SCENARIOS: dict[str, tuple[str, str, Callable[[Sim], None]]] = {
    "nominal": ("--", "clean loop and one textbook adaptation", sc_nominal),
    "capture-timeout": ("P1 v_capture_timeout", "Monitor never records the frame", sc_capture_timeout),
    "detection-timeout": ("P1 v_detection_timeout", "Analysis never completes", sc_detection_timeout),
    "window-overflow": ("P2 v_window_overflow", "entropy window longer than configured", sc_window_overflow),
    "entropy-range": ("P2 v_entropy_range", "a sample outside [0,1]", sc_entropy_range),
    "avg-range": ("P2 v_avg_range", "running average above 1.0", sc_avg_range),
    "avg-mismatch": ("P2 v_avg_mismatch", "average is not the mean of the window", sc_avg_mismatch),
    "brightness": ("P3 w_brightness_range", "frame far too dark (warning only)", sc_brightness),
    "anomaly-unsound": ("P4 v_anomaly_unsound", "anomaly on a partial, healthy window", sc_anomaly_unsound),
    "missed-anomaly": ("P5 v_missed_anomaly", "degraded past the grace period, no anomaly", sc_missed_anomaly),
    "overlapping-anomaly": ("P6 v_overlapping_anomaly", "second anomaly inside an open episode", sc_overlapping_anomaly),
    "unlegitimated-execution": ("P6 v_unlegitimated_execution", "actuation with no validated plan", sc_unlegitimated_execution),
    "unsolicited-plan": ("P6 v_unsolicited_plan", "plan produced with no anomaly open", sc_unsolicited_plan),
    "unsolicited-verdict": ("P6 v_unsolicited_verdict", "verdict with no plan to judge", sc_unsolicited_verdict),
    "plan-timeout": ("P7 v_plan_timeout", "Plan misses its deadline", sc_plan_timeout),
    "legit-timeout": ("P7 v_legit_timeout", "Legitimate misses its deadline", sc_legit_timeout),
    "execute-timeout": ("P7 v_execute_timeout", "Execute misses its deadline", sc_execute_timeout),
    "episode-timeout": ("P8 v_episode_timeout", "episode outlives 120s, every step in time", sc_episode_timeout),
    "candidate-self-swap": ("P9 v_candidate_self_swap", "candidate equals the active model", sc_candidate_self_swap),
    "candidate-unknown": ("P9 v_candidate_unknown", "candidate model id out of range", sc_candidate_unknown),
    "unjustified-accept": ("P10 v_unjustified_accept", "validated against its own rule", sc_unjustified_accept),
    "unjustified-reject": ("P10 v_unjustified_reject", "rejected against its own rule", sc_unjustified_reject),
    "verdict-flag-mismatch": ("P10 v_verdict_flag_mismatch", "is_legit disagrees with the event", sc_verdict_flag_mismatch),
    "replan-budget": ("P11 v_replan_budget", "a fourth rejection in one episode", sc_replan_budget),
    "counter-leak": ("P11 v_counter_leak", "episode opens with a leaked counter", sc_counter_leak),
    "execute-mismatch": ("P12 v_execute_mismatch", "actuated model is not the validated one", sc_execute_mismatch),
    "post-reset": ("P13 v_post_reset", "state not cleared after the swap", sc_post_reset),
    "ineffective-adaptation": ("P14 v_ineffective_adaptation", "still degraded after the swap", sc_ineffective_adaptation),
    "thrashing": ("P15 v_thrashing", "three swaps inside the stability window", sc_thrashing),
    "ping-pong": ("P15 v_ping_pong", "model A -> B -> A (also trips v_thrashing)", sc_ping_pong),
}


# Streams a scenario raises *in addition* to the one named in its catalogue
# entry, because the specification cannot separate them.
EXTRA_STREAMS: dict[str, tuple[str, ...]] = {
    # An A -> B -> A oscillation is necessarily three swaps inside the same
    # stability window, so v_thrashing cannot be avoided here. See README.
    "ping-pong": ("v_thrashing",),
}


def expected_streams(name: str) -> list[str]:
    """The violation streams a scenario is supposed to raise.

    Taken from the stream name embedded in the catalogue entry (e.g.
    "P11 v_counter_leak"), plus any co-firing stream listed in EXTRA_STREAMS.
    `nominal` names none, which is the point: anything it raises is a bug.
    """
    prop = SCENARIOS[name][0]
    streams = [s for s in re.findall(r"[vw]_[a-z_]+", prop) if s in STREAM_KIND]
    for stream in EXTRA_STREAMS.get(name, ()):
        if stream not in streams:
            streams.append(stream)
    return streams


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Drive the MAPLE-K trustworthiness checker into named violations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--scenario",
        default="nominal",
        help="scenario name, or 'all' to run every one in sequence (see --list)",
    )
    parser.add_argument("--list", action="store_true", help="list scenarios and exit")
    parser.add_argument(
        "--redis-host", default="127.0.0.1", help="never 'localhost' on Windows"
    )
    parser.add_argument("--redis-port", type=int, default=6379)
    parser.add_argument("--event-db", type=int, default=0, help="db for pub/sub events and t")
    parser.add_argument(
        "--knowledge-db", type=int, default=2, help="db for knowledge keys (see maplek_input.json5)"
    )
    parser.add_argument(
        "--tick-step", type=float, default=1.0,
        help="virtual seconds each plain tick advances",
    )
    parser.add_argument(
        "--tick-interval", type=float, default=0.35,
        help="real seconds between ticks; the checker evaluates one tick per "
        "published t, so publishing faster than it can keep up just builds a backlog",
    )
    parser.add_argument(
        "--clock-first",
        action="store_true",
        help="publish t before the knowledge/event of a step instead of after",
    )
    parser.add_argument(
        "--enable-keyspace-events",
        action="store_true",
        help="set notify-keyspace-events=KA on the server if knowledge "
        "notifications are off (without it you only get a warning)",
    )
    parser.add_argument(
        "--settle-ticks",
        type=int,
        default=2,
        help="ticks to wait after writing knowledge before publishing the event "
        "it explains (0 reproduces the race)",
    )
    parser.add_argument(
        "--start-t",
        type=float,
        default=None,
        help="force the virtual clock to start here instead of resuming from "
        "redis; use 0 only when restarting the checker as well",
    )
    parser.add_argument(
        "--hold-ticks",
        type=int,
        default=6,
        help="ticks to sit in an injected state before clearing it, so a "
        "level-triggered violation stays true long enough for the dashboard to "
        "sample it (0 restores the old one-tick behaviour). Has no effect on "
        "edge-triggered streams -- those are one tick by construction",
    )
    parser.add_argument(
        "--no-observe",
        action="store_true",
        help="do not subscribe to the checker's output channels; without this "
        "the driver reports which expected violations actually fired",
    )
    parser.add_argument(
        "--drain",
        type=float,
        default=1.5,
        help="real seconds to wait after a scenario for the checker's verdicts "
        "to arrive before judging it",
    )
    parser.add_argument(
        "--repeat", type=int, default=1, help="run the scenario N times (0 = forever)"
    )
    parser.add_argument("--dry-run", action="store_true", help="print the timeline, touch no redis")
    parser.add_argument("--quiet", action="store_true", help="no per-step log")
    return parser.parse_args()


def print_list() -> None:
    width = max(len(name) for name in SCENARIOS)
    print(f"{'scenario'.ljust(width)}  property                       kind   what it does")
    print(f"{'-' * width}  {'-' * 29}  {'-' * 5}  {'-' * 44}")
    for name, (prop, description, _) in SCENARIOS.items():
        streams = expected_streams(name)
        kinds = {STREAM_KIND[stream] for stream in streams}
        kind = "/".join(sorted(kinds)) if kinds else "--"
        print(f"{name.ljust(width)}  {prop.ljust(29)}  {kind.ljust(5)}  {description}")
    print()
    print(
        "kind=edge means the stream is true for exactly one tick -- real, but "
        "the dashboard shows the current value and will usually never sample "
        "it. Trust the RESULT lines this driver prints, not the bars."
    )


def announce(sim: Sim, name: str, streams: list[str]) -> None:
    """Say, before anything is injected, exactly what should turn red."""
    prop, description, _ = SCENARIOS[name]
    sim.say(f"=== {name} -- expect {prop} ({description})")
    if not streams:
        sim.say("    EXPECT no violation at all -- this is the healthy baseline")
        return
    for stream in streams:
        kind = STREAM_KIND[stream]
        if kind == LEVEL:
            ticks = sim.args.hold_ticks + 1
            detail = (
                f"true for ~{ticks} ticks "
                f"(~{ticks * sim.args.tick_interval:.1f}s) -- watch the bar"
            )
        else:
            detail = (
                "true for ONE tick only -- the dashboard shows the current "
                "value and will almost certainly miss it"
            )
        sim.say(f"    EXPECT {stream} -> true  [{kind}] {detail}")


def report(sim: Sim, name: str, streams: list[str], before, after) -> str:
    """Compare what the checker actually published against what we expected."""
    if before is None or after is None:
        sim.say(f"    RESULT {name}: not observed (--no-observe / --dry-run)")
        return "unobserved"

    delta = {stream: after[stream] - before[stream] for stream in set(after) | set(before)}
    fired = [stream for stream in streams if delta.get(stream, 0) > 0]
    missing = [stream for stream in streams if delta.get(stream, 0) == 0]
    collateral = sorted(
        stream for stream, count in delta.items() if count > 0 and stream not in streams
    )

    if not streams:
        verdict = "PASS" if not collateral else "FAIL"
    elif not missing:
        verdict = "PASS"
    elif fired:
        verdict = "PARTIAL"
    else:
        verdict = "FAIL"

    sim.say(f"    RESULT {name}: {verdict}")
    if fired:
        sim.say(f"      fired      {', '.join(fired)}")
    if missing:
        sim.say(f"      MISSING    {', '.join(missing)}")
    if collateral:
        sim.say(f"      collateral {', '.join(collateral)}")
    return verdict


def run_once(sim: Sim, name: str) -> str:
    _, _, scenario = SCENARIOS[name]
    streams = expected_streams(name)
    announce(sim, name, streams)

    before = sim.observer.snapshot() if sim.observer else None
    scenario(sim)
    sim.recover()
    if sim.observer is not None:
        # The checker is a tick or two behind us; let its verdicts land before
        # we decide whether they arrived.
        time.sleep(sim.args.drain)
    after = sim.observer.snapshot() if sim.observer else None
    return report(sim, name, streams, before, after)


def main() -> None:
    args = parse_args()

    if args.list:
        print_list()
        return

    names = list(SCENARIOS) if args.scenario == "all" else [args.scenario]
    unknown = [name for name in names if name not in SCENARIOS]
    if unknown:
        raise SystemExit(
            f"unknown scenario {unknown[0]!r}; run --list for the catalogue"
        )

    sim = Sim(args)
    for key in BASELINE_KNOWLEDGE:
        sim.write(key)
    sim.tick()

    target = "(dry run)" if args.dry_run else (
        f"{args.redis_host}:{args.redis_port} events=db{args.event_db} "
        f"knowledge=db{args.knowledge_db}"
    )
    print(
        f"[sim] {target}  tick_step={args.tick_step}s every {args.tick_interval}s  "
        f"t starts at {sim.vt:.1f}s  (Ctrl+C to stop)",
        flush=True,
    )

    verdicts: dict[str, str] = {}
    iteration = 0
    try:
        while args.repeat == 0 or iteration < args.repeat:
            for name in names:
                verdicts[name] = run_once(sim, name)
                if len(names) > 1:
                    # Outrun the 600s stability window so P15 does not bleed
                    # from one scenario into the next.
                    sim.advance(STABILITY_WINDOW + 20.0, "cooldown between scenarios")
            iteration += 1
    except KeyboardInterrupt:
        print("\n[sim] stopped", flush=True)
    finally:
        print_summary(sim, verdicts)


def print_summary(sim: Sim, verdicts: dict[str, str]) -> None:
    """The bottom line: what was expected, and what the checker actually said."""
    if not verdicts:
        return

    print("\n=== summary " + "=" * 58, flush=True)
    width = max(len(name) for name in verdicts)
    for name, verdict in verdicts.items():
        streams = ", ".join(expected_streams(name)) or "(none expected)"
        print(f"  {verdict.ljust(9)} {name.ljust(width)}  {streams}", flush=True)

    if sim.observer is None:
        return

    if sim.observer.messages == 0:
        print(
            "\n[sim] the checker published NOTHING on any violation channel: it is"
            "\n[sim] not running, was started without --redis-output, is pointed at"
            "\n[sim] another redis, or has silently wedged (README section 7). No"
            "\n[sim] scenario can pass until that is fixed.",
            flush=True,
        )
        return

    evaluated = sim.observer.messages / max(len(STREAM_KIND), 1)
    if sim.ticks and evaluated < sim.ticks * 0.8:
        print(
            f"\n[sim] the checker evaluated roughly {evaluated:.0f} of the {sim.ticks} ticks"
            "\n[sim] published. It is running behind, so verdicts may land against the"
            "\n[sim] wrong scenario. Raise --tick-interval, raise --drain, or run a"
            "\n[sim] release build of the checker.",
            flush=True,
        )

    failed = [name for name, verdict in verdicts.items() if verdict in {"FAIL", "PARTIAL"}]
    if failed:
        print(
            f"\n[sim] {len(failed)} scenario(s) did not raise everything expected."
            "\n[sim] Check notify-keyspace-events (every knowledge-driven property"
            "\n[sim] goes quiet without it) and that t is still monotonic for this"
            "\n[sim] checker process.",
            flush=True,
        )
    else:
        print(
            f"\n[sim] all {len(verdicts)} scenario(s) raised their target stream.",
            flush=True,
        )


if __name__ == "__main__":
    main()
