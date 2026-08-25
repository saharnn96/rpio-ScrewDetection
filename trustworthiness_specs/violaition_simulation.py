#!/usr/bin/env python3
"""Fault-injecting driver for the MAPLE-K trustworthiness checker.

Plays the role of the whole managing system: publishes the 8 MAPLE-K event
channels, the `t` clock and the 9 knowledge keys that `maple_k.dsrv` reads,
following the real protocol -- except that each scenario bends it in exactly
one place so a named property fires.

This is the *input* side of the checker. `test_tc_dashboard_redis.py` fakes
the *output* side; this one makes a real checker produce those outputs.

    python simulation.py --list
    python simulation.py --scenario counter-leak
    python simulation.py --scenario all

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
import time
import uuid
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

        self.write("replanning_counter", count=0)
        self.write("in_planning", in_planning=False)
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
        self.advance(plan_gap)
        self.write("candidate_model", model_id=3)
        self.emit("plan_generated")
        self.advance(verdict_gap)
        self.write(
            "legit_result",
            is_legit=False,
            candidate_model_id=3,
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
    sim.emit("observation_recorded")  # late, clears the flag
    sim.emit("detection_completed")


def sc_detection_timeout(sim: Sim) -> None:
    sim.emit("sensor_data_received", detection_type="screw")
    sim.emit("observation_recorded")
    sim.advance(25.0, "Analysis is wedged (detection_deadline is 20s)")
    sim.emit("detection_completed")


def sc_window_overflow(sim: Sim) -> None:
    sim.set_window([0.2] * (ENTROPY_WINDOW_SIZE + 2))
    sim.advance(3.0, "window holds 12 samples, configured size is 10")
    sim.set_window([])


def sc_entropy_range(sim: Sim) -> None:
    # 1.7 is outside [0,1]; the other samples keep the mean low so nothing else fires.
    sim.set_window([0.1] * 9 + [1.7])
    sim.advance(3.0, "one sample outside [0,1]")
    sim.set_window([])


def sc_avg_range(sim: Sim) -> None:
    # Empty window: len 0 disables the mismatch check, isolating v_avg_range.
    sim.set_window([], avg=1.5)
    sim.advance(3.0, "running average 1.5 with an empty window")
    sim.set_window([])


def sc_avg_mismatch(sim: Sim) -> None:
    sim.write("in_planning", in_planning=True)  # suppress P5 while we sit here
    sim.set_window([0.2] * ENTROPY_WINDOW_SIZE, avg=0.9)
    sim.advance(3.0, "published average 0.9, true mean 0.2")
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
    sim.set_window([])


def sc_unlegitimated_execution(sim: Sim) -> None:
    # Everything else about the execution is consistent -- only the protocol is broken.
    sim.write(
        "legit_result",
        is_legit=True,
        candidate_model_id=3,
        candidate_avg_entropy=0.20,
        current_avg_entropy=0.55,
    )
    sim.finish_episode(3)  # straight from idle: no anomaly, no plan, no verdict
    sim.fill_window(0.20)


def sc_unsolicited_plan(sim: Sim) -> None:
    sim.write("candidate_model", model_id=3)
    sim.emit("plan_generated")  # phase is idle, nobody asked for a plan
    sim.advance(2.0)


def sc_plan_timeout(sim: Sim) -> None:
    sim.open_episode()
    sim.advance(13.0, "Plan never responds (plan_deadline is 10s)")


def sc_episode_timeout(sim: Sim) -> None:
    # Keep the per-step clocks fresh by cycling plan/reject, so only the
    # whole-episode bound is exceeded. Two rejections stay inside the budget.
    sim.open_episode()
    sim.reject_cycle(plan_gap=3.0, verdict_gap=50.0, counter=1)
    sim.reject_cycle(plan_gap=3.0, verdict_gap=50.0, counter=2)
    sim.advance(3.0)
    sim.write("candidate_model", model_id=3)
    sim.emit("plan_generated")
    sim.advance(20.0, "episode past 120s while each step is still inside its deadline")


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
    sim.open_episode()
    sim.advance(2.0)
    sim.write("candidate_model", model_id=3)
    sim.emit("plan_generated")
    sim.advance(3.0)
    # Candidate is worse than the incumbent and over threshold: the rule says reject.
    sim.write(
        "legit_result",
        is_legit=True,
        candidate_model_id=3,
        candidate_avg_entropy=0.60,
        current_avg_entropy=0.50,
    )
    sim.emit("plan_validated")
    sim.advance(3.0)
    sim.finish_episode(3)
    sim.fill_window(0.20)


def sc_verdict_flag_mismatch(sim: Sim) -> None:
    sim.open_episode()
    sim.advance(2.0)
    sim.write("candidate_model", model_id=3)
    sim.emit("plan_generated")
    sim.advance(3.0)
    # The rule accepts and the event says validated, but the published flag says no.
    sim.write(
        "legit_result",
        is_legit=False,
        candidate_model_id=3,
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
    sim.open_episode()
    sim.advance(2.0)
    sim.write("candidate_model", model_id=3)
    sim.emit("plan_generated")
    sim.advance(3.0)
    sim.write(
        "legit_result",
        is_legit=True,
        candidate_model_id=3,
        candidate_avg_entropy=0.20,
        current_avg_entropy=0.55,
    )
    sim.emit("plan_validated")
    sim.advance(3.0)
    sim.finish_episode(3, action_model=MAX_MODEL_ID)  # actuated a different model
    sim.fill_window(0.20)


def sc_post_reset(sim: Sim) -> None:
    sim.open_episode()
    sim.advance(2.0)
    sim.write("candidate_model", model_id=3)
    sim.emit("plan_generated")
    sim.advance(3.0)
    sim.write(
        "legit_result",
        is_legit=True,
        candidate_model_id=3,
        candidate_avg_entropy=0.20,
        current_avg_entropy=0.55,
    )
    sim.emit("plan_validated")
    sim.advance(3.0)
    # Swap done, but the window was never cleared and the counter still holds 1.
    sim.finish_episode(3, counter=1, window=[0.2, 0.2, 0.2])
    sim.write("replanning_counter", count=0)
    sim.fill_window(0.20)


def sc_ineffective_adaptation(sim: Sim) -> None:
    sim.open_episode()
    sim.advance(2.0)
    sim.write("candidate_model", model_id=3)
    sim.emit("plan_generated")
    sim.advance(3.0)
    sim.write(
        "legit_result",
        is_legit=True,
        candidate_model_id=3,
        candidate_avg_entropy=0.20,
        current_avg_entropy=0.55,
    )
    sim.emit("plan_validated")
    sim.advance(3.0)
    sim.finish_episode(3)
    # The new model is no better: the refilled window is still over threshold.
    sim.fill_window(0.55)
    sim.set_window([])


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
    "unlegitimated-execution": ("P6 v_unlegitimated_execution", "actuation with no validated plan", sc_unlegitimated_execution),
    "unsolicited-plan": ("P6 v_unsolicited_plan", "plan produced with no anomaly open", sc_unsolicited_plan),
    "plan-timeout": ("P7 v_plan_timeout", "Plan misses its deadline", sc_plan_timeout),
    "episode-timeout": ("P8 v_episode_timeout", "episode outlives 120s, every step in time", sc_episode_timeout),
    "candidate-self-swap": ("P9 v_candidate_self_swap", "candidate equals the active model", sc_candidate_self_swap),
    "candidate-unknown": ("P9 v_candidate_unknown", "candidate model id out of range", sc_candidate_unknown),
    "unjustified-accept": ("P10 v_unjustified_accept", "validated against its own rule", sc_unjustified_accept),
    "verdict-flag-mismatch": ("P10 v_verdict_flag_mismatch", "is_legit disagrees with the event", sc_verdict_flag_mismatch),
    "replan-budget": ("P11 v_replan_budget", "a fourth rejection in one episode", sc_replan_budget),
    "counter-leak": ("P11 v_counter_leak", "episode opens with a leaked counter", sc_counter_leak),
    "execute-mismatch": ("P12 v_execute_mismatch", "actuated model is not the validated one", sc_execute_mismatch),
    "post-reset": ("P13 v_post_reset", "state not cleared after the swap", sc_post_reset),
    "ineffective-adaptation": ("P14 v_ineffective_adaptation", "still degraded after the swap", sc_ineffective_adaptation),
    "thrashing": ("P15 v_thrashing", "three swaps inside the stability window", sc_thrashing),
    "ping-pong": ("P15 v_ping_pong", "model A -> B -> A (also trips v_thrashing)", sc_ping_pong),
}


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
        "--repeat", type=int, default=1, help="run the scenario N times (0 = forever)"
    )
    parser.add_argument("--dry-run", action="store_true", help="print the timeline, touch no redis")
    parser.add_argument("--quiet", action="store_true", help="no per-step log")
    return parser.parse_args()


def print_list() -> None:
    width = max(len(name) for name in SCENARIOS)
    print(f"{'scenario'.ljust(width)}  property                       what it does")
    print(f"{'-' * width}  {'-' * 29}  {'-' * 44}")
    for name, (prop, description, _) in SCENARIOS.items():
        print(f"{name.ljust(width)}  {prop.ljust(29)}  {description}")


def run_once(sim: Sim, name: str) -> None:
    prop, description, scenario = SCENARIOS[name]
    sim.say(f"=== {name} -- expect {prop} ({description})")
    scenario(sim)
    sim.recover()


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

    iteration = 0
    try:
        while args.repeat == 0 or iteration < args.repeat:
            for name in names:
                run_once(sim, name)
                if len(names) > 1:
                    # Outrun the 600s stability window so P15 does not bleed
                    # from one scenario into the next.
                    sim.advance(STABILITY_WINDOW + 20.0, "cooldown between scenarios")
            iteration += 1
    except KeyboardInterrupt:
        print("\n[sim] stopped", flush=True)


if __name__ == "__main__":
    main()
