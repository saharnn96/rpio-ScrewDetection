"""Knowledge / message classes that flow through the MAPLE-K knowledge store.

Contract (from reading `rpclpy/node.py` for real deployment):
  * Each class MUST expose a `name` class attribute. Real rpclpy uses it as
    the redis key for that message type: `knowledge.write(cls.name, pickled)`.
  * Instances are pickled on write, so all fields must be pickle-friendly
    (str/int/float/bool/list/dict/tuple/None). Do NOT put numpy arrays, cv2
    frames or torch tensors here - pass a file path or a plain list instead.
    The camera frame never travels through the knowledge store, only its
    on-disk path (FrameRef.frame_path).
  * Real rpclpy also stamps `uid` and `timestamp` onto each written instance;
    we don't declare them because they are populated at write time.

The offline `local_bus.py` shim also keys by `cls.name`, so the same message
definitions work identically in both modes.
"""


class FrameRef:
    """Pointer to the image captured this cycle (written by Monitor)."""
    name = "FrameRef"

    def __init__(self):
        self.frame_path = None
        self.timestamp = None
        self.detection_type = None
        self.tcp_pose = None
        self.brightness = None


class Detections:
    """Latest detection summary (written by Analysis). JSON-friendly."""
    name = "Detections"

    def __init__(self):
        self.items = []
        self.screw_entropy = None


class EntropyHistory:
    """Rolling window of recent screw/noscrew entropies."""
    name = "EntropyHistory"

    def __init__(self):
        self.values = []


class RunningAvgEntropy:
    """Running average over EntropyHistory (drives the anomaly decision)."""
    name = "RunningAvgEntropy"

    def __init__(self):
        self.value = None


class ActiveModel:
    """The model currently deployed for detection."""
    name = "ActiveModel"

    def __init__(self):
        self.model_id = 1


class CandidateModel:
    """A proposed replacement model produced by Plan, validated by Legitimate."""
    name = "CandidateModel"

    def __init__(self):
        self.model_id = None
        self.reason = None


class InPlanning:
    """True while an adaptation (Plan -> Legitimate -> Execute) is in flight."""
    name = "InPlanning"

    def __init__(self):
        self.in_planning = False


class ReplanningCounter:
    """How many times Legitimate has rejected a candidate this episode."""
    name = "ReplanningCounter"

    def __init__(self):
        self.count = 0


class LegitResult:
    """Outcome of Legitimate's validation (the /isLegit data payload)."""
    name = "LegitResult"

    def __init__(self):
        self.is_legit = False
        self.candidate_model_id = None
        self.candidate_avg_entropy = None
        self.current_avg_entropy = None


class ActionCommand:
    """The action Execute applied to the live system (the /action_command payload)."""
    name = "ActionCommand"

    def __init__(self):
        self.command = None
        self.model_id = None
        self.timestamp = None
