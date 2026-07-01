"""rpclpy port of the screw-detection MAPLE-K loop.

This is the `_rpio` ("rpclpy I/O") version of `python/screwSegmentation.py`.
The original ran the whole MAPLE-K loop inside one synchronous XMLRPC method
(`take_new_image_and_detect`). Here each phase is its own rpclpy `Node` that
reacts to events and communicates through the shared knowledge store:

    SensorData --(Monitor)--> new_data --(Analysis)--> anomaly
        --(Plan)--> new_plan --(Legitimate)--> isLegit --(Execute)--> action_command
                                          \--> anomaly  (re-plan loop)

Mapping to the original code's region comments:
    MONITOR    region  ->  Monitor.monitor      (tniad_monitor / capture)
    ANALYZE    region  ->  Analysis.analysis    (run_uq + uq_analysis + entropy)
    PLAN       region  ->  Plan.planner         (the `if entropy > thr` stub)
    LEGITIMATE region  ->  Legitimate.legitimizer (re-test new model on rolling imgs)
    KNOWLEDGE  region  ->  Execute.executer      (log + apply the model swap)

The heavy/hardware-specific work (camera, robot, model, UQ) lives behind the
injected `ScrewDetectionCore`, so the same node code runs against real hardware
or against the simulator in `simulation.py`.

It tries to import the real rpclpy; if that is unavailable it falls back to the
in-process `local_bus` shim so the simulation runs with zero extra setup.
"""

import enum
import os
import shutil
import time
from collections import namedtuple

# --- rpclpy with offline fallback ----------------------------------------
try:
    from rpclpy.node import Node
    from rpclpy.utils import timeit_callback
    from rpclpy.DashboardApp import run_dashboard
    _USING_REAL_RPCLPY = True
except Exception:  # rpclpy not installed / no redis -> use the local shim
    from local_bus import Node, timeit_callback, run_dashboard
    _USING_REAL_RPCLPY = False

from messages import (
    FrameRef, Detections, EntropyHistory, RunningAvgEntropy, ActiveModel,
    CandidateModel, InPlanning, ReplanningCounter, LegitResult, ActionCommand,
)


# ---------------------------------------------------------------------------
# Small self-contained copies of the enums/namedtuple from the original file.
# (Re-declared here so this module never imports the hardware-heavy original.)
# ---------------------------------------------------------------------------
class DetectionClasses(enum.Enum):
    HOLDER = 0
    SCREW = 1
    NOSCREW = 2


class DetectionModels(enum.Enum):
    PC_CLOSEUP = 1
    SCREW_FIXTURE = 2
    SCREEN_AND_SCREEN_FRAME = 3
    PC_TOPVIEW = 4


DetectionResult = namedtuple(
    "DetectionResult", ["label", "box", "score", "mask", "entropy"]
)


# ---------------------------------------------------------------------------
# Detection core - all hardware/model access goes through injected adapters.
# ---------------------------------------------------------------------------
class ScrewDetectionCore:
    """Holds the camera, robot and detector and provides the primitives the
    MAPLE-K nodes call. Hardware is injected so the simulator can swap in fakes.

    Required adapter interfaces:
        camera.get_color_image() -> np.ndarray (BGR)
        rtde.get_tcp_pose()      -> dict(x,y,z,rx,ry,rz)
        detector.detect(image, model_id) -> (list[DetectionResult], screw_entropy|None)
    """

    def __init__(self, camera, rtde, detector, out_dir="./_sim_out",
                 entropy_window_size=10, entropy_threshold=0.5,
                 candidate_model_id=2, max_replans=3):
        self.camera = camera
        self.rtde = rtde
        self.detector = detector

        self.out_dir = out_dir
        self.rolling_dir = os.path.join(out_dir, "entropy_rolling_average_images")
        self.capture_dir = os.path.join(out_dir, "real_time_detection_images")
        os.makedirs(self.rolling_dir, exist_ok=True)
        os.makedirs(self.capture_dir, exist_ok=True)

        self.entropy_window_size = entropy_window_size
        self.entropy_threshold = entropy_threshold
        self.candidate_model_id = candidate_model_id
        self.max_replans = max_replans

        self.knowledge_log_path = os.path.join(out_dir, "knowledge_log.csv")

    # --- import cv2 lazily so the module imports even without OpenCV -------
    @staticmethod
    def _cv2():
        import cv2
        return cv2

    # --- MONITOR primitive -------------------------------------------------
    def capture(self, detection_type):
        """Grab a frame, persist it, return a JSON-friendly reference dict."""
        cv2 = self._cv2()
        frame = self.camera.get_color_image()
        timestamp = time.strftime("%Y%m%d_%H%M%S") + f"_{int(time.time()*1000)%1000:03d}"
        frame_path = os.path.join(self.capture_dir, f"new_image_{detection_type}_{timestamp}.jpg")
        cv2.imwrite(frame_path, frame)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        brightness = float(gray.mean())
        return {
            "frame_path": frame_path,
            "timestamp": timestamp,
            "brightness": brightness,
        }

    # --- ANALYZE primitive -------------------------------------------------
    def detect(self, frame_path, model_id):
        """Run detection + UQ on a saved frame. Returns (detections, screw_entropy)."""
        cv2 = self._cv2()
        image = cv2.imread(frame_path)
        return self.detector.detect(image, model_id)

    def append_rolling_image(self, frame_path):
        """Keep only the last `entropy_window_size` frames in the rolling dir."""
        dst = os.path.join(self.rolling_dir, os.path.basename(frame_path))
        shutil.copyfile(frame_path, dst)
        images = sorted(os.listdir(self.rolling_dir))
        while len(images) > self.entropy_window_size:
            os.remove(os.path.join(self.rolling_dir, images.pop(0)))

    def list_rolling_images(self):
        return [os.path.join(self.rolling_dir, n) for n in sorted(os.listdir(self.rolling_dir))]

    # --- KNOWLEDGE primitive ----------------------------------------------
    def log_knowledge(self, timestamp, frame_ref, detections):
        import csv
        new_file = not os.path.exists(self.knowledge_log_path)
        with open(self.knowledge_log_path, "a", newline="") as f:
            writer = csv.writer(f)
            if new_file:
                writer.writerow(["timestamp", "brightness", "active_model",
                                 "num_detections", "screw_entropy"])
            screw_entropy = next((d["entropy"] for d in detections
                                  if d["label"] in (DetectionClasses.SCREW.value,
                                                    DetectionClasses.NOSCREW.value)
                                  and d["entropy"] is not None), None)
            writer.writerow([timestamp, f"{frame_ref.get('brightness', 0):.2f}",
                             self.detector.model_id, len(detections),
                             f"{screw_entropy:.4f}" if screw_entropy is not None else ""])


# ---------------------------------------------------------------------------
# A module-level handle to the shared core. main()/simulation.py set this so
# every node operates on the same camera/model instance (the "run nodes in one
# process, share the model by reference" approach).
# ---------------------------------------------------------------------------
CORE: ScrewDetectionCore = None


def set_core(core):
    global CORE
    CORE = core


# ===========================================================================
# MAPLE-K NODES
# ===========================================================================
class Monitor(Node):
    """MONITOR: capture a frame and announce new data."""

    def __init__(self, config=None, verbose=True):
        super().__init__(config=config, verbose=verbose)
        self._name = "Monitor"
        self.logger.info("Monitor instantiated")
        # Seed shared knowledge.
        self.write_knowledge(self._fresh(ActiveModel))
        self.write_knowledge(self._fresh(EntropyHistory))
        self.write_knowledge(self._fresh(RunningAvgEntropy))
        self.write_knowledge(self._fresh(InPlanning))
        self.write_knowledge(self._fresh(ReplanningCounter))

    @staticmethod
    def _fresh(cls):
        return cls()

    @timeit_callback
    def monitor(self, msg):
        import json
        data = json.loads(msg) if isinstance(msg, str) else (msg or {})
        detection_type = data.get("detection_type", "pc_screen")
        tcp_pose = data.get("tcp_pose") or CORE.rtde.get_tcp_pose()

        ref = CORE.capture(detection_type)
        frame_ref = FrameRef()
        frame_ref.frame_path = ref["frame_path"]
        frame_ref.timestamp = ref["timestamp"]
        frame_ref.detection_type = detection_type
        frame_ref.tcp_pose = tcp_pose
        frame_ref.brightness = ref["brightness"]
        self.write_knowledge(frame_ref)

        self.logger.info("MONITOR: captured %s (brightness=%.1f)",
                         os.path.basename(ref["frame_path"]), ref["brightness"])
        self.publish_event(event_key="new_data")

    def register_callbacks(self):
        self.register_event_callback(event_key="SensorData", callback=self.monitor)


class Analysis(Node):
    """ANALYZE: run detection + UQ, update the entropy window, flag anomalies."""

    def __init__(self, config=None, verbose=True):
        super().__init__(config=config, verbose=verbose)
        self._name = "Analysis"
        self.logger.info("Analysis instantiated")

    @timeit_callback
    def analysis(self, msg):
        frame_ref = self.read_knowledge(FrameRef)
        active = self.read_knowledge(ActiveModel)

        detections, screw_entropy = CORE.detect(frame_ref.frame_path, active.model_id)

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

        if screw_entropy is None:
            self.logger.info("ANALYZE: no single screw/noscrew detection; skipping entropy update")
            return

        # Update rolling entropy window + running average.
        hist = self.read_knowledge(EntropyHistory)
        hist.values.append(screw_entropy)
        if len(hist.values) > CORE.entropy_window_size:
            hist.values.pop(0)
        self.write_knowledge(hist)

        avg = sum(hist.values) / len(hist.values)
        run_avg = RunningAvgEntropy()
        run_avg.value = avg
        self.write_knowledge(run_avg)

        CORE.append_rolling_image(frame_ref.frame_path)
        self.logger.info("ANALYZE: screw_entropy=%.3f  running_avg=%.3f (window=%d)",
                         screw_entropy, avg, len(hist.values))

        # Anomaly decision: sustained high uncertainty AND not already adapting.
        in_planning = self.read_knowledge(InPlanning)
        if (avg > CORE.entropy_threshold and not in_planning.in_planning
                and len(hist.values) >= CORE.entropy_window_size):
            in_planning.in_planning = True
            self.write_knowledge(in_planning)
            self.logger.warning("ANALYZE: anomaly! avg %.3f > threshold %.3f -> trigger Plan",
                                avg, CORE.entropy_threshold)
            self.publish_event(event_key="anomaly")

    def register_callbacks(self):
        self.register_event_callback(event_key="new_data", callback=self.analysis)


class Plan(Node):
    """PLAN: propose a different model/light configuration to reduce uncertainty."""

    def __init__(self, config=None, verbose=True):
        super().__init__(config=config, verbose=verbose)
        self._name = "Plan"
        self.logger.info("Plan instantiated")

    @timeit_callback
    def planner(self, msg):
        active = self.read_knowledge(ActiveModel)
        replans = self.read_knowledge(ReplanningCounter)

        # Simple policy: swap to the configured candidate model. On re-plan,
        # keep proposing it (a richer policy would pick from a model zoo here).
        candidate = CandidateModel()
        candidate.model_id = CORE.candidate_model_id
        candidate.reason = (f"running_avg entropy exceeded {CORE.entropy_threshold}; "
                            f"swap from model {active.model_id} "
                            f"(replan #{replans.count})")
        self.write_knowledge(candidate)

        self.logger.info("PLAN: proposing candidate model %s (%s)",
                         candidate.model_id, candidate.reason)
        self.publish_event(event_key="new_plan")

    def register_callbacks(self):
        self.register_event_callback(event_key="anomaly", callback=self.planner)


class Legitimate(Node):
    """LEGITIMATE: re-test the candidate model on the last N frames before committing."""

    def __init__(self, config=None, verbose=True):
        super().__init__(config=config, verbose=verbose)
        self._name = "Legitimate"
        self.logger.info("Legitimate instantiated")

    @timeit_callback
    def legitimizer(self, msg):
        candidate = self.read_knowledge(CandidateModel)
        run_avg = self.read_knowledge(RunningAvgEntropy)
        replans = self.read_knowledge(ReplanningCounter)

        # Re-run the candidate model over the rolling-average images and average
        # its screw/noscrew entropy (the original LEGITIMATE region logic).
        entropies = []
        for img_path in CORE.list_rolling_images():
            _, e = CORE.detect(img_path, candidate.model_id)
            entropies.append(e if e is not None else 1.0)
        candidate_avg = sum(entropies) / len(entropies) if entropies else None

        result = LegitResult()
        result.candidate_model_id = candidate.model_id
        result.candidate_avg_entropy = candidate_avg
        result.current_avg_entropy = run_avg.value

        accept = (candidate_avg is not None and run_avg.value is not None
                  and candidate_avg < run_avg.value)

        if accept:
            result.is_legit = True
            self.write_knowledge(result)
            # Reset the replanning counter on success.
            replans.count = 0
            self.write_knowledge(replans)
            self.logger.info("LEGITIMATE: candidate avg %.3f < current %.3f -> ACCEPT, trigger Execute",
                             candidate_avg, run_avg.value)
            self.publish_event(event_key="isLegit")
        else:
            result.is_legit = False
            self.write_knowledge(result)
            replans.count += 1
            self.write_knowledge(replans)
            ca = f"{candidate_avg:.3f}" if candidate_avg is not None else "n/a"
            cur = f"{run_avg.value:.3f}" if run_avg.value is not None else "n/a"
            if replans.count <= CORE.max_replans:
                self.logger.warning("LEGITIMATE: reject (cand %s >= cur %s) -> re-plan #%d",
                                    ca, cur, replans.count)
                self.publish_event(event_key="anomaly")
            else:
                # Give up adapting this episode; clear the in-planning flag.
                in_planning = self.read_knowledge(InPlanning)
                in_planning.in_planning = False
                self.write_knowledge(in_planning)
                self.logger.error("LEGITIMATE: reject and max replans (%d) reached -> abort adaptation",
                                  CORE.max_replans)

    def register_callbacks(self):
        self.register_event_callback(event_key="new_plan", callback=self.legitimizer)


class Execute(Node):
    """EXECUTE/KNOWLEDGE: commit the validated model swap and log the cycle."""

    def __init__(self, config=None, verbose=True):
        super().__init__(config=config, verbose=verbose)
        self._name = "Execute"
        self.logger.info("Execute instantiated")

    @timeit_callback
    def executer(self, msg):
        result = self.read_knowledge(LegitResult)
        if not result or not result.is_legit:
            return

        # Apply the swap to the live system (core + detector).
        CORE.detector.model_id = result.candidate_model_id
        active = ActiveModel()
        active.model_id = result.candidate_model_id
        self.write_knowledge(active)

        cmd = ActionCommand()
        cmd.command = "swap_model"
        cmd.model_id = result.candidate_model_id
        cmd.timestamp = time.strftime("%Y%m%d_%H%M%S")
        self.write_knowledge(cmd)

        # Reset the entropy window so the new model is judged on fresh data.
        fresh_hist = EntropyHistory()
        self.write_knowledge(fresh_hist)
        run_avg = RunningAvgEntropy()
        self.write_knowledge(run_avg)

        # Clear the in-planning latch -> monitoring resumes normally.
        in_planning = InPlanning()
        in_planning.in_planning = False
        self.write_knowledge(in_planning)

        # KNOWLEDGE: persist the cycle.
        frame_ref = self.read_knowledge(FrameRef)
        dets = self.read_knowledge(Detections)
        CORE.log_knowledge(cmd.timestamp,
                           {"brightness": getattr(frame_ref, "brightness", 0)},
                           dets.items if dets else [])

        self.logger.info("EXECUTE: swapped to model %s and reset entropy window",
                         result.candidate_model_id)
        self.publish_event(event_key="action_command")

    def register_callbacks(self):
        self.register_event_callback(event_key="isLegit", callback=self.executer)


# ===========================================================================
# Wiring helpers
# ===========================================================================
def build_nodes(config):
    """Instantiate, register and start all five MAPLE-K nodes."""
    nodes = []
    for cls, key in [(Monitor, "Monitor_Config"), (Analysis, "Analysis_Config"),
                     (Plan, "Plan_Config"), (Legitimate, "Legitimate_Config"),
                     (Execute, "Execute_Config")]:
        node = cls(config.get(key) if isinstance(config, dict) else None)
        node.register_callbacks()
        node.start()
        nodes.append(node)
    return nodes


def main():
    """Entry point for REAL hardware (camera + robot + YOLO/UQ).

    The simulation uses `simulation.py` instead. This path lazily builds real
    adapters so importing this module never requires pyrealsense2/ultralytics.
    """
    import yaml
    cfg_path = os.path.join(os.path.dirname(__file__), "config.yaml")
    with open(cfg_path) as f:
        config = yaml.safe_load(f)

    # Real adapters would be constructed here, e.g.:
    #   from real_adapters import RealCamera, RealRTDE, RealDetector
    #   core = ScrewDetectionCore(RealCamera(...), RealRTDE(...), RealDetector(...))
    raise NotImplementedError(
        "main() is the real-hardware entry point. Build RealCamera/RealRTDE/"
        "RealDetector adapters and call set_core(...) + build_nodes(config). "
        "For the offline demo run `python simulation.py`."
    )


if __name__ == "__main__":
    main()
