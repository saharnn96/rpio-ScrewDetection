"""MANAGING SYSTEM - the five MAPLE-K rpclpy nodes.

Monitor / Analysis / Plan / Legitimate / Execute adapt the managed system
(`managed_system/`). Nodes never talk to hardware. They only:
  * read/write knowledge objects (messages.py),
  * publish/subscribe events,
  * call the probe/effector methods on `managed_system.core.CORE`.

Adaptation POLICY (entropy threshold, window size, candidate model, replan
budget) lives in `managing_system/config.yaml` under `Adaptation_Config`;
`build_nodes()` applies it on top of the DEFAULT_ADAPTATION class attributes.

Real rpclpy is imported if available; otherwise the local in-process shim from
`local_bus.py` is used. `Node`, `timeit_callback` and `run_dashboard` are
re-exported so downstream files (`simulation.py`, `real_main.py`) never have
to repeat the try/except.
"""

import os
import time

# --- rpclpy with offline fallback (imported once, re-exported for others) ---
# We only commit to the real rpclpy if BOTH the package imports AND a redis
# server is actually reachable, since rpclpy hard-depends on redis and crashes
# at the first knowledge write otherwise. Falling back cleanly means
# `python simulation.py` works with zero setup even if rpclpy happens to be
# installed on the machine.
def _try_real_rpclpy():
    try:
        from rpclpy.node import Node as _Node
        from rpclpy.utils import timeit_callback as _tc
        from rpclpy.DashboardApp import run_dashboard as _rd
        import redis as _redis
        _redis.Redis(host="localhost", port=6379,
                     socket_connect_timeout=0.5).ping()
        return _Node, _tc, _rd, True
    except Exception:
        from managing_system.local_bus import (
            Node as _Node, timeit_callback as _tc, run_dashboard as _rd,
        )
        return _Node, _tc, _rd, False

Node, timeit_callback, run_dashboard, USING_REAL_RPCLPY = _try_real_rpclpy()

from managed_system import core as ss
from managing_system.messages import (
    FrameRef, Detections, EntropyHistory, RunningAvgEntropy, ActiveModel,
    CandidateModel, InPlanning, ReplanningCounter, LegitResult, ActionCommand,
)

# Adaptation policy defaults; overridden per deployment by the
# `Adaptation_Config` section of managing_system/config.yaml (see build_nodes).
DEFAULT_ADAPTATION = {
    "entropy_window_size": 10,
    "entropy_threshold": 0.5,
    "candidate_model_id": 2,
    "max_replans": 3,
}


# ===========================================================================
# MONITOR
# ===========================================================================
class Monitor(Node):
    """MONITOR: capture a frame from the injected camera and announce new data."""

    def __init__(self, config=None, verbose=True):
        super().__init__(config=config, verbose=verbose)
        self._name = "Monitor"
        self.logger.info("Monitor instantiated")
        # Seed shared knowledge with default instances.
        self.write_knowledge(ActiveModel())
        self.write_knowledge(EntropyHistory())
        self.write_knowledge(RunningAvgEntropy())
        self.write_knowledge(InPlanning())
        self.write_knowledge(ReplanningCounter())

    @timeit_callback
    def monitor(self, msg):
        import json
        data = json.loads(msg) if isinstance(msg, str) else (msg or {})
        detection_type = data.get("detection_type", "pc_screen")
        tcp_pose = data.get("tcp_pose") or ss.CORE.rtde.get_tcp_pose()

        ref = ss.CORE.capture(detection_type)
        frame_ref = FrameRef()
        frame_ref.frame_path = ref["frame_path"]
        frame_ref.timestamp = ref["timestamp"]
        frame_ref.detection_type = detection_type
        frame_ref.tcp_pose = tcp_pose
        frame_ref.brightness = ref["brightness"]
        self.write_knowledge(frame_ref)

        self.logger.info("MONITOR: captured %s (brightness=%.1f)",
                         os.path.basename(ref["frame_path"]), ref["brightness"])
        self.publish_event(event_key="observation_recorded")

    def register_callbacks(self):
        self.register_event_callback(event_key="sensor_data_received", callback=self.monitor)


# ===========================================================================
# ANALYZE
# ===========================================================================
class Analysis(Node):
    """ANALYZE: run UQ, update the entropy window, flag anomalies."""

    entropy_window_size = DEFAULT_ADAPTATION["entropy_window_size"]
    entropy_threshold = DEFAULT_ADAPTATION["entropy_threshold"]

    def __init__(self, config=None, verbose=True):
        super().__init__(config=config, verbose=verbose)
        self._name = "Analysis"
        self.logger.info("Analysis instantiated")

    @timeit_callback
    def analysis(self, msg):
        frame_ref = self.read_knowledge(FrameRef)
        active = self.read_knowledge(ActiveModel)

        # The task (detection_type) picks the base model; MAPLE-K only owns the
        # model choice for the adaptive (closeup) task.
        detection_type = getattr(frame_ref, "detection_type", None)
        model_id = ss.CORE.resolve_model_id(detection_type, active.model_id)
        detections, screw_entropy = ss.CORE.detect(frame_ref.frame_path, model_id)

        # Store a JSON-friendly detection summary.
        det_msg = Detections()
        det_msg.items = [
            {"label": d.label,
             "box": [float(x) for x in d.box] if d.box is not None else None,
             "score": float(d.score) if d.score is not None else None,
             "entropy": float(d.entropy) if d.entropy is not None else None}
            for d in detections
        ]
        det_msg.screw_entropy = float(screw_entropy) if screw_entropy is not None else None
        self.write_knowledge(det_msg)

        # Detections are in the knowledge store - anyone blocked on this cycle
        # (e.g. the XMLRPC bridge serving the UR pendant) can proceed now.
        # The entropy/adaptation logic below is MAPLE-K-internal and must not
        # gate the pendant's synchronous call.
        self.publish_event(event_key="detection_completed")

        # Non-adaptive tasks (topview / screen / fixture) detect and stop here -
        # no entropy window, no anomaly, no model swap.
        if not ss.CORE.is_adaptive(detection_type):
            self.logger.info("ANALYZE: detection_type=%s is non-adaptive (model %s); "
                             "skipping entropy/anomaly logic", detection_type, model_id)
            return

        if screw_entropy is None:
            self.logger.info("ANALYZE: no single screw/noscrew detection; skipping entropy update")
            return

        hist = self.read_knowledge(EntropyHistory)
        hist.values.append(screw_entropy)
        if len(hist.values) > self.entropy_window_size:
            hist.values.pop(0)
        self.write_knowledge(hist)

        avg = sum(hist.values) / len(hist.values)
        run_avg = RunningAvgEntropy()
        run_avg.value = avg
        self.write_knowledge(run_avg)

        ss.CORE.append_rolling_image(frame_ref.frame_path, self.entropy_window_size)
        self.logger.info("ANALYZE: screw_entropy=%.3f  running_avg=%.3f (window=%d)",
                         screw_entropy, avg, len(hist.values))

        in_planning = self.read_knowledge(InPlanning)
        if (avg > self.entropy_threshold and not in_planning.in_planning
                and len(hist.values) >= self.entropy_window_size):
            in_planning.in_planning = True
            self.write_knowledge(in_planning)
            self.logger.warning("ANALYZE: anomaly! avg %.3f > threshold %.3f -> trigger Plan",
                                avg, self.entropy_threshold)
            self.publish_event(event_key="anomaly_detected")

    def register_callbacks(self):
        self.register_event_callback(event_key="observation_recorded", callback=self.analysis)


# ===========================================================================
# PLAN
# ===========================================================================
class Plan(Node):
    """PLAN: propose a different model / light configuration."""

    entropy_threshold = DEFAULT_ADAPTATION["entropy_threshold"]
    candidate_model_id = DEFAULT_ADAPTATION["candidate_model_id"]

    def __init__(self, config=None, verbose=True):
        super().__init__(config=config, verbose=verbose)
        self._name = "Plan"
        self.logger.info("Plan instantiated")

    @timeit_callback
    def planner(self, msg):
        active = self.read_knowledge(ActiveModel)
        replans = self.read_knowledge(ReplanningCounter)

        candidate = CandidateModel()
        candidate.model_id = self.candidate_model_id
        candidate.reason = (f"running_avg entropy exceeded {self.entropy_threshold}; "
                            f"swap from model {active.model_id} "
                            f"(replan #{replans.count})")
        self.write_knowledge(candidate)

        self.logger.info("PLAN: proposing candidate model %s (%s)",
                         candidate.model_id, candidate.reason)
        self.publish_event(event_key="plan_generated")

    def register_callbacks(self):
        self.register_event_callback(event_key="anomaly_detected", callback=self.planner)
        self.register_event_callback(event_key="plan_rejected", callback=self.planner)


# ===========================================================================
# LEGITIMATE (the trustworthiness gate)
# ===========================================================================
class Legitimate(Node):
    """LEGITIMATE: re-test the candidate on the last N frames before committing."""

    max_replans = DEFAULT_ADAPTATION["max_replans"]
    entropy_threshold = DEFAULT_ADAPTATION["entropy_threshold"]

    def __init__(self, config=None, verbose=True):
        super().__init__(config=config, verbose=verbose)
        self._name = "Legitimate"
        self.logger.info("Legitimate instantiated")

    @timeit_callback
    def legitimizer(self, msg):
        candidate = self.read_knowledge(CandidateModel)
        run_avg = self.read_knowledge(RunningAvgEntropy)
        replans = self.read_knowledge(ReplanningCounter)

        # Re-run the candidate over the rolling images from disk.
        entropies = []
        for img_path in ss.CORE.list_rolling_images():
            _, e = ss.CORE.detect(img_path, candidate.model_id)
            entropies.append(e if e is not None else 1.0)
        candidate_avg = sum(entropies) / len(entropies) if entropies else None

        result = LegitResult()
        result.candidate_model_id = candidate.model_id
        result.candidate_avg_entropy = candidate_avg
        result.current_avg_entropy = run_avg.value

        # Relaxed gate: the candidate does not have to strictly beat the
        # incumbent. It is legit if it improves on the current average OR its
        # own average is below the anomaly threshold - i.e. deploying it would
        # clear the anomaly that started this planning cycle.
        beats_current = (candidate_avg is not None and run_avg.value is not None
                         and candidate_avg < run_avg.value)
        clears_threshold = (candidate_avg is not None
                            and candidate_avg < self.entropy_threshold)
        accept = beats_current or clears_threshold

        cur = f"{run_avg.value:.3f}" if run_avg.value is not None else "n/a"
        if accept:
            result.is_legit = True
            self.write_knowledge(result)
            replans.count = 0
            self.write_knowledge(replans)
            reason = ("beats current avg" if beats_current
                      else "below anomaly threshold")
            self.logger.info("LEGITIMATE: candidate avg %.3f (current %s, "
                             "threshold %.3f) -> ACCEPT (%s)",
                             candidate_avg, cur, self.entropy_threshold, reason)
            self.publish_event(event_key="plan_validated")
        else:
            result.is_legit = False
            self.write_knowledge(result)
            replans.count += 1
            self.write_knowledge(replans)
            ca = f"{candidate_avg:.3f}" if candidate_avg is not None else "n/a"
            if replans.count <= self.max_replans:
                self.logger.warning("LEGITIMATE: reject (cand %s >= cur %s and >= "
                                    "threshold %.3f) -> re-plan #%d",
                                    ca, cur, self.entropy_threshold, replans.count)
                self.publish_event(event_key="plan_rejected")
            else:
                in_planning = self.read_knowledge(InPlanning)
                in_planning.in_planning = False
                self.write_knowledge(in_planning)
                self.logger.error("LEGITIMATE: reject and max replans (%d) reached -> abort",
                                  self.max_replans)

    def register_callbacks(self):
        self.register_event_callback(event_key="plan_generated", callback=self.legitimizer)


# ===========================================================================
# EXECUTE (+ Knowledge write-back)
# ===========================================================================
class Execute(Node):
    """EXECUTE: commit the validated model swap and log the cycle."""

    def __init__(self, config=None, verbose=True):
        super().__init__(config=config, verbose=verbose)
        self._name = "Execute"
        self.logger.info("Execute instantiated")

    @timeit_callback
    def executer(self, msg):
        result = self.read_knowledge(LegitResult)
        if not result or not result.is_legit:
            return

        # Apply the swap through the managed system's effector.
        ss.CORE.swap_model(result.candidate_model_id)
        active = ActiveModel()
        active.model_id = result.candidate_model_id
        self.write_knowledge(active)

        cmd = ActionCommand()
        cmd.command = "swap_model"
        cmd.model_id = result.candidate_model_id
        cmd.timestamp = time.strftime("%Y%m%d_%H%M%S")
        self.write_knowledge(cmd)

        # Reset the entropy window so the new model gets a clean judgment.
        self.write_knowledge(EntropyHistory())
        self.write_knowledge(RunningAvgEntropy())

        in_planning = InPlanning()
        in_planning.in_planning = False
        self.write_knowledge(in_planning)

        # KNOWLEDGE write-back.
        frame_ref = self.read_knowledge(FrameRef)
        dets = self.read_knowledge(Detections)
        ss.CORE.log_knowledge(cmd.timestamp,
                              {"brightness": getattr(frame_ref, "brightness", 0)},
                              dets.items if dets else [])

        self.logger.info("EXECUTE: swapped to model %s and reset entropy window",
                         result.candidate_model_id)
        self.publish_event(event_key="plan_executed")

    def register_callbacks(self):
        self.register_event_callback(event_key="plan_validated", callback=self.executer)


# ===========================================================================
# Wiring helper - used by both simulation.py and real_main.py.
# ===========================================================================
def build_nodes(config):
    """Instantiate, register and start all five MAPLE-K nodes.

    `config` is the managing-system config (managing_system/config.yaml):
    per-node sections plus the `Adaptation_Config` policy block, which is
    applied over DEFAULT_ADAPTATION onto the nodes that use each value.
    """
    config = config if isinstance(config, dict) else {}
    adaptation = {**DEFAULT_ADAPTATION, **(config.get("Adaptation_Config") or {})}
    unknown = set(adaptation) - set(DEFAULT_ADAPTATION)
    if unknown:
        raise ValueError(f"Unknown Adaptation_Config keys: {sorted(unknown)}; "
                         f"valid: {sorted(DEFAULT_ADAPTATION)}")

    nodes = []
    for cls, key in [(Monitor, "Monitor_Config"), (Analysis, "Analysis_Config"),
                     (Plan, "Plan_Config"), (Legitimate, "Legitimate_Config"),
                     (Execute, "Execute_Config")]:
        node = cls(config.get(key))
        for param in DEFAULT_ADAPTATION:
            if hasattr(type(node), param):
                setattr(node, param, adaptation[param])
        node.register_callbacks()
        node.start()
        nodes.append(node)
    return nodes
