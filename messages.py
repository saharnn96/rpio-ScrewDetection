"""Knowledge / message classes that flow through the MAPLE-K knowledge store.

Each class is a plain attribute holder. The knowledge store keys objects by
their class name, so `read_knowledge(EntropyHistory)` returns the most recently
written `EntropyHistory` instance.

Design notes for porting to real rpclpy + redis:
  * Keep these objects small and JSON-friendly (str/int/float/bool/list/dict).
    Do NOT put numpy arrays, cv2 frames or torch tensors in here - pass a file
    path or a plain list instead. The camera frame itself never travels through
    the knowledge store; only its on-disk path (FrameRef.frame_path) does.
"""


class FrameRef:
    """Pointer to the image captured this cycle (written by Monitor)."""
    def __init__(self):
        self.frame_path = None        # str: path to the saved frame on disk
        self.timestamp = None         # str
        self.detection_type = None    # str: "pc_screen" | "screw_fixture" | ...
        self.tcp_pose = None          # list[float]: [x,y,z,rx,ry,rz]
        self.brightness = None        # float: mean grayscale intensity 0..255


class Detections:
    """Latest detection summary (written by Analysis). JSON-friendly."""
    def __init__(self):
        self.items = []               # list[dict]: {label, box, score, entropy}
        self.screw_entropy = None     # float | None: entropy of the single screw/noscrew


class EntropyHistory:
    """Rolling window of recent screw/noscrew entropies (the original deque)."""
    def __init__(self):
        self.values = []              # list[float]


class RunningAvgEntropy:
    """Running average over EntropyHistory (drives the anomaly decision)."""
    def __init__(self):
        self.value = None             # float | None


class ActiveModel:
    """The model currently deployed for detection."""
    def __init__(self):
        self.model_id = 1             # int


class CandidateModel:
    """A proposed replacement model produced by Plan, validated by Legitimate."""
    def __init__(self):
        self.model_id = None          # int | None
        self.reason = None            # str


class InPlanning:
    """True while an adaptation (Plan -> Legitimate -> Execute) is in flight."""
    def __init__(self):
        self.in_planning = False      # bool


class ReplanningCounter:
    """How many times Legitimate has rejected a candidate this episode."""
    def __init__(self):
        self.count = 0                # int


class LegitResult:
    """Outcome of Legitimate's validation (the /isLegit data payload)."""
    def __init__(self):
        self.is_legit = False         # bool
        self.candidate_model_id = None
        self.candidate_avg_entropy = None
        self.current_avg_entropy = None


class ActionCommand:
    """The action Execute applied to the live system (the /action_command payload)."""
    def __init__(self):
        self.command = None           # str, e.g. "swap_model"
        self.model_id = None          # int
        self.timestamp = None         # str
