# Trustworthy Self-Adaptation for Vision-Guided Robotic Disassembly: An Industrial Experience Report from Laptop Refurbishment

**Authors:** Sahar Nasimi Nezhad¹, Thomas David Wright², Malaika Din Hashmi³, Francois Picard³, Paul De Meulenaere¹
¹University of Antwerp · ²Aarhus University · ³Danish Technological Institute (DTI)

**Funding:** This work was carried out within the RoboSAPIENS project.

> *Draft status: Markdown working draft. Sections are complete-prose where the
> underlying evidence is solid; sections resting on data or decisions we
> haven't confirmed yet are marked `TODO`. Section 7 (Lessons Learned) is a
> first pass — flagged for a deeper follow-up pass per author feedback.*

---

## Abstract

Image-detection models are widely used in manufacturing. In our case study, a
robot assembles and disassembles the screens of refurbished laptops, and the
screws that hold the chassis together are located using YOLO-based object
detectors. In practice a single YOLO model does not perform well across the
range of lighting conditions the cell encounters over a shift, since a model
tuned for one lighting regime often performs badly under another. To handle
this we make the system self-adaptive, so that it switches between several
image-detection models trained for different lighting conditions as
conditions change on the shop floor. The system is built on the RoboSAPIENS
framework, introduced by the RoboSAPIENS project, which structures the
adaptation as a MAPE-K feedback loop. Using this real industrial case study
we show how a trustworthy MAPE-K loop can switch between image-detection
models while keeping the overall system trustworthy before, during, and after
each adaptation. We report on the deployment of this self-adaptive system on
the real robotic laptop-refurbishment cell and share the lessons learned in
the process of adapting it.

---

## 1. Introduction

Laptop refurbishment is a manual, fixture-driven, and lighting-sensitive
disassembly task: a technician (or, in our case, a UR5 robot) must remove the
screws holding the back cover and bezel before the screen can be swapped.
Automating the vision step — finding screws, empty screw holes, and holders,
then locating the screen and its frame — sounds like a solved object-detection
problem until it is run for a full shift: sunlight through a workshop window,
overhead lighting schedules, and even the laptop's own color and finish shift
the effective illumination of the fixture, and a YOLO detector calibrated for
one lighting condition quietly produces less confident, then wrong,
detections in another.

The common industrial answer — swap in a model variant trained for the new
lighting condition — is simple to *state* and hard to trust in production:
who verifies that the swap only happens when it should, that the candidate
model is actually better, that the swap completes before the robot commits to
a wrong coordinate, and that the system doesn't oscillate between two models
indefinitely? This is the trustworthy-adaptation problem, and it is exactly
what the RoboSAPIENS reference architecture targets: separate the mechanism
that reacts (a MAPLE-K adaptation loop) from an independent means of holding
that mechanism accountable (a runtime-verified trustworthiness case).

This paper reports the industrial experience of building and deploying such
a system for DTI's robotic laptop-refurbishment cell. Our contributions are:

1. A concrete instantiation of the RoboSAPIENS managed/managing split for a
   real vision-guided disassembly cell, preserving the pre-existing robot
   program's interface so the adaptation loop could be introduced without
   re-validating the robot side of the deployment.
2. A **two-layer trust design**: an in-loop *Legitimate* stage that
   empirically re-tests a candidate model before any actuation, plus an
   out-of-loop, independently developed runtime-verification checker that
   audits the whole adaptation protocol against a specification of what
   "trustworthy" means for this loop.
3. A graduated, three-tier deployment strategy (simulation → light-box rig
   with a real camera → the real cell) built on the same core logic, and the
   engineering lessons from moving between those tiers, including a live
   field incident.
4. An honest accounting of what we can and cannot yet claim about long-run
   field performance of the adaptive loop, grounded in a two-month
   pre-adaptation production log that motivates why the problem is real and
   frequent.

---

## 2. Industrial Context and Use Case

DTI operates a robotic cell for laptop refurbishment as part of a
circular-economy/e-waste-reduction program: rather than discarding laptops
with a broken or degraded screen, a UR5 robot disassembles the chassis,
technicians (or a further automated step) replace the screen, and the
refurbished device is reassembled. Screws are the gatekeeper of this process:
the robot cannot proceed to lift the back cover, and eventually reach the
screen and its bezel, until it has correctly identified every screw
location, distinguished a screw from an empty screw hole ("noscrew"), and
found the fixture holder used to stage removed screws.

The vision system is organized around several camera/model roles matching
distinct stages of the disassembly sequence:

| Role | What it detects |
|---|---|
| `pc_topview` | Coarse top-down localization of the laptop and its screw layout |
| `pc_closeup` | Close-range screw / no-screw / holder classification for the actual unscrewing action |
| `screw_fixture` | The tray/fixture where removed screws are staged |
| `screen_and_screen_frame` | Segmentation of the laptop screen and its bezel/frame, ahead of the screen-replacement step |

Each role's model is a YOLO detector (or, for the screen/frame role, a
segmentation model whose masks are reduced to oriented bounding boxes). The
underlying object classes are `HOLDER`, `NOSCREW`, and `SCREW`; their
predicted order in the model output is load-bearing — it is what tells the
robot controller which coordinate is a screw to remove versus a hole already
empty — so any relabeling bug sends the screwdriver to the wrong location. We
mention this because it recurred as a concrete regression risk throughout
the project (see §5) and is exactly the kind of defect a trustworthiness
layer should be able to catch structurally rather than by chance.

### 2.1 The lighting problem, quantified

Before this project introduced adaptive model-swapping, the cell's original
control software (`ScrewSegmentation.py`) already *computed* a lighting
classification for every detection cycle — but the branch that would have
acted on it was dead code. We recovered its production log
(`knowledge_log.csv`), covering **3,568 detection cycles across roughly two
months (2026‑05‑26 to 2026‑07‑28)**, and used it purely as a *baseline*
characterizing how often lighting regimes actually change in this cell — not
as a measurement of the new adaptive system, which had not yet accumulated a
comparable operating history at the time of writing.

Lighting-regime distribution across all logged cycles:

| Regime | Cycles | Share |
|---|---|---|
| MEDIUM | 1,882 | 52.8% |
| BRIGHT | 973 | 27.3% |
| DARK | 694 | 19.4% |
| NONE (undetermined) | 18 | 0.5% |

Cycles by camera/model role:

| Role | Cycles |
|---|---|
| `pc_closeup` | 2,021 |
| `screw_fixture` | 1,109 |
| `pc_topview` | 223 |
| `screen_and_screen_frame` | 214 |

The takeaway we draw from this baseline is not a precise failure rate (the
original system had no swap mechanism to fail or succeed) but that lighting
regime is not a rare edge condition on this floor: across two months of
normal operation, the cell spent substantial, comparable amounts of time in
all three regimes, with a near-6:1 swing between the least and most common.
A vision pipeline hard-tuned to one regime is, on this evidence, guaranteed
to spend a large fraction of its operating life outside its comfort zone.

*(TODO: confirm with DTI whether this window covers a representative mix of
shifts/seasons, or was concentrated in one period — affects how strongly we
can claim "representative of steady-state operation.")*

---

## 3. Background

### 3.1 MAPE-K and the RoboSAPIENS reference architecture

MAPE-K (Monitor–Analyze–Plan–Execute, over shared Knowledge) is the standard
decomposition for self-adaptive systems. RoboSAPIENS extends this to
robotic systems with an explicit separation between a **managed system**
(the application-level primitives: sensing, perception, actuation) and a
**managing system** that observes the managed system through probes and
adapts it through effectors — deliberately keeping "what the robot does"
and "how the robot decides to change how it does it" as separately
reasoned-about, separately testable concerns. Our variant follows this split
one-to-one, naming the loop stages Monitor–Analyze–Plan–**Legitimate**–Execute
(MAPLE-K): the extra *Legitimate* stage is RoboSAPIENS's answer to "how do
you know a proposed adaptation should actually be trusted before you act on
it," discussed further in §3.2 and §4.4.

### 3.2 Two notions of trust: in-loop legitimation vs. out-of-loop verification

RoboSAPIENS treats trustworthiness as something that needs *both* a fast,
local, in-the-loop check and a slower, global, independent one:

- **Legitimate** is a stage inside the adaptation loop. When Plan proposes a
  candidate model, Legitimate re-runs it against recently observed frames
  (the "rolling window") before Execute is allowed to commit the swap. It
  answers a narrow, empirical question: *is this specific candidate actually
  better than what's running now, right now, on real recent data?* It is
  fast (bounded by re-inference cost) and can only ever veto or approve one
  proposed action at a time.
- The **TC (trustworthiness case) checker** is a passive, out-of-loop runtime
  verification monitor. It does not see "a candidate" — it sees the entire
  adaptation protocol unfold as a stream of events and knowledge updates, and
  checks *global* properties no single in-loop stage is positioned to check:
  that no actuation ever happens without a validated plan, that deadlines are
  respected, that a proposed candidate isn't just the currently active model
  in disguise, that a rejection budget isn't silently leaking across
  episodes, that the system doesn't thrash between two models. It answers a
  structural question: *is the loop, as a whole, behaving the way its
  designers claimed it would?*

These are complementary, not redundant. Legitimate can approve a swap that
is locally sound but still part of a globally pathological pattern (e.g., an
A→B→A oscillation, each individual swap locally justified); the TC checker
cannot, by itself, decide whether a specific candidate model is good — it can
only tell whether the *protocol* around that decision was followed. Framing
"trustworthy adaptation" for this project meant designing for both.

The TC checker's specification (§5's `trustworthiness_specs/`) is written in
a **LOLA-inspired stream-based specification language** — it follows LOLA's
stream-equation style (bounded past references, `if`-`then`-`else`,
boolean/arithmetic combinators over declared input/output streams) but is
not LOLA proper; it is the dialect used by the checker actually deployed for
this project, developed independently by a separate runtime-verification
team against a stream-contract we authored (see §5). We are careful in this
paper to describe it as LOLA-*style*, not to claim LOLA compliance.

*(TODO: name/cite the actual specification language/tool once we confirm
what's publishable about the checker's provenance — the RV team may have a
preferred citation.)*

---

## 4. Solution Architecture

### 4.1 Layered design

The architecture is organized in layers so that the *same* adaptation logic
can run against progressively more realistic hardware without being
rewritten:

```raw
        managing_system/            (MAPLE-K nodes, adaptation policy,
                                      event bus, unchanged across tiers)
               │  probes / effectors
        managed_system/core.py      (application primitives: capture,
                                      detect, coordinate transforms —
                                      unchanged across tiers)
               │
        managed_system/adapters/    (the ONLY layer that changes)
          ├── sim.py        synthetic camera + deterministic entropy model
          ├── lightbox.py   real RealSense camera, mocked robot
          └── real.py       real camera + real UR robot + legacy RPC bridge
               │
        composition roots (entry points)
          simulation.py · lightbox_main.py · real_main.py
```

Only the adapter layer and the composition root change between simulation,
the light-box rig, and the shop floor; the managing system's five nodes, the
adaptation policy (entropy window, threshold, candidate selection, replan
budget), and the TC checker's contract against the event/knowledge bus are
identical in all three tiers. This is what made the graduated deployment
strategy in §5 possible without three divergent implementations to keep in
sync — and, as importantly, it means anything the TC checker verifies in
simulation is verifying the *same* protocol code path that later runs against
the real robot, not a simulation-only stand-in.

### 4.2 Managed system

`ScrewDetectionCore` exposes the probe/effector surface the managing system
drives: probes `capture`, `detect`, `append_rolling_image`,
`list_rolling_images`, `log_knowledge`; effector `swap_model`. Hardware
specifics (robot IP, camera calibration, model weight paths) live in the
managed system's own configuration, deliberately separate from the
managing system's adaptation policy — a decision that paid off directly: the
same `Adaptation_Config` (window size, threshold, candidate id, replan
budget) drives Simulated, Lightbox and Real detectors alike, and a policy
change (e.g., a new threshold) never touches deployment-fact configuration
or vice versa.

### 4.3 Managing system: the MAPLE-K nodes

Five nodes, communicating over an event bus (redis pub/sub for events,
pickled knowledge objects keyed by class name) using the `rpclpy` framework:

| Phase | Node | Reads | Writes | Publishes |
|---|---|---|---|---|
| Monitor | `Monitor.monitor` | `ActiveModel` | `FrameRef` | `observation_recorded` |
| Analyze | `Analysis.analysis` | `FrameRef`, `ActiveModel`, `EntropyHistory`, `InPlanning` | `Detections`, `EntropyHistory`, `RunningAvgEntropy`, `InPlanning` | `detection_completed` (every cycle), `anomaly_detected` (only when the running-average detection entropy crosses threshold and no adaptation is already in flight) |
| Plan | `Plan.planner` | `ActiveModel`, `ReplanningCounter` | `CandidateModel` | `plan_generated` |
| Legitimate | `Legitimate.legitimizer` | `CandidateModel`, `RunningAvgEntropy`, rolling images | `LegitResult`, `ReplanningCounter` | `plan_validated` / `plan_rejected` (bounded by a replan budget) |
| Execute | `Execute.executer` | `LegitResult`, `FrameRef`, `Detections` | `ActionCommand`, `ActiveModel`, `EntropyHistory` (reset), `InPlanning=False` | `plan_executed` |

A structural property this design relies on for the TC checker to be sound
at all — every node writes its knowledge *before* publishing the
corresponding event — means the checker can always sample knowledge on the
tick of an event and see exactly the values that event refers to, without a
race between "the event arrived" and "the knowledge caught up."

### 4.4 Legacy-compatible integration

The shop-floor cell already ran a UR pendant program talking to a Python
server over a synchronous XML-RPC contract on a fixed port and method
surface (`take_new_image_and_detect`, `get_detected_object_coords`, exposure
and calibration calls). Rather than asking DTI to accept a new robot program
alongside a new adaptation loop — two changes, two sources of risk, in one
deployment — `bridge.py` reproduces the exact legacy RPC surface and maps
each synchronous pendant call onto the (asynchronous) MAPLE-K event bus,
blocking the pendant call until the corresponding `detection_completed`
event arrives (or timing out with a Fault, logged with a correlation id to
`xmlrpc_errors.log` for diagnosis). This is deliberately outside both
`managed_system` and `managing_system`: it is neither an adapter (it is
inbound, not outbound) nor a loop node — it exists purely to make the
adaptation loop invisible to the existing robot program. In an industrial
setting where the robot program itself is a validated, change-controlled
artifact, this was as important to the deployment's acceptance as anything
in the adaptation logic itself.

---

## 5. Deployment Journey: From Simulation to the Shop Floor

### 5.1 Simulation tier

`simulation.py` drives the loop against a synthetic camera whose background
brightness deliberately dims partway through a run, and a synthetic detector
whose entropy is a pure, deterministic function of frame brightness relative
to each model's "optimum" brightness. This gave us, before any hardware was
involved, a reproducible end-to-end trace of the intended behavior: running
average entropy climbs past threshold → `anomaly_detected` → Plan proposes a
candidate → Legitimate re-tests it against the same (saved) rolling frames
Analyze saw and accepts it → Execute commits the swap → entropy drops on
subsequent ticks. Because the entropy function is deterministic, Legitimate's
offline re-test is guaranteed consistent with Analyze's live measurement,
letting us validate the *protocol* in isolation from any ML model quality
question.

### 5.2 Light-box tier

An educational light-box rig — a real Intel RealSense camera, no robot yet
(robot commands mocked) — let us validate the adapter boundary against real
sensor noise and a real (if fixture-scale) YOLO model, with a human able to
change the box lighting by hand mid-run to trigger adaptation on demand. This
tier caught issues invisible in simulation: the light-box models' class
schema didn't match the field pipeline's (`{screw, hole}` vs.
`{holder, noscrew, screw}`), requiring an explicit label-translation layer;
and the field pipeline's uncertainty-quantification code path depends on a
patched YOLO fork not installed on this machine, so the light-box detector
falls back to a pseudo-entropy measure — a limitation we keep visible rather
than silently papering over, since it means light-box trials validate the
*adaptation protocol* faithfully but not the *exact* entropy numbers the
field system would compute.

### 5.3 Field tier and a live incident

On the real cell, an early integration pass surfaced a concrete production
defect: the pendant's `take_new_image_and_detect` call for the
`screen_and_screen_frame` object type timed out, because the ported
Analysis path threw `NotImplementedError` for that model id and never
published `detection_completed` — the segmentation-model detection path
(masks → area-ratio filtering → oriented bounding boxes for the screen and
its frame) simply hadn't been carried over from the legacy code yet. The
correlation id logged alongside the timeout in `xmlrpc_errors.log` let us
trace the failure directly to the missing code path from the dashboard's
node logs, without instrumenting anything new. We ported the missing
detection path, re-validated it against the light-box tier first, and only
then re-deployed to the cell.

We flag this not as a footnote but as the clearest evidence in this report
for the value of the graduated, three-tier strategy in §4.1: the defect was
caught during controlled field integration testing rather than during
unattended production operation, specifically *because* the same adaptation
logic could be exercised on the light-box rig before being trusted against
the shop-floor cell again — and the fix could be re-validated on that same
rig before a second field attempt.

---

## 6. Evaluation

We separate what we can support with evidence from what remains ongoing
work, deliberately, since an industrial experience report is only useful if
its claims are calibrated to its evidence.

### 6.1 Motivating baseline

§2.1's two-month, 3,568-cycle log from the pre-adaptive system establishes
that lighting-regime variability is frequent and material on this specific
floor — the necessary condition for an adaptive approach to be worth its
complexity in the first place.

### 6.2 Protocol validation

§5.1-5.3 establish, across three tiers of increasing hardware realism, that
the MAPLE-K protocol executes its intended trace end-to-end (anomaly →
candidate → legitimation → execution → reset) and that the layered
architecture lets defects be caught before unattended field operation.

### 6.3 What we have not yet measured

We do not yet have a long-horizon field operating history *of the adaptive
loop itself* comparable to the two-month baseline in §2.1 — the field
deployment is recent relative to that baseline. We therefore do not claim,
in this draft, a measured reduction in mis-detection rate, downtime, or
manual-recalibration incidents attributable to adaptation; nor have we yet
observed the TC checker's two anticipated findings (§6.4) fire against live
production traffic, only against the code paths that produce them. These are
the natural next data points once the field system accumulates comparable
operating history, and we intend to report them in a revision rather than
project them here.

### 6.4 A note on the TC checker's prospective findings

Two of the specification's fifteen properties are expected to be violated by
the system exactly as currently built — genuine latent defects the
specification writing process itself surfaced, independent of any monitor
run: (1) Plan's candidate selection is a fixed id regardless of which model
is currently active, so a "swap" can be proposed to the model already
running, which cannot possibly clear the anomaly that triggered it; and (2)
Legitimate's rejection path never resets the replanning counter on an
aborted episode, so the *next* episode inherits an already-consumed replan
budget. We list these not as embarrassments but as the paper's strongest
concrete evidence that writing the trustworthiness specification *before*
extensive field operation has independent value: both were found by the act
of formalizing "what should never happen," not by observing a failure in
production.

---

## 7. Lessons Learned

*(First pass — flagged by the authors for a deeper follow-up revision with
more concrete, DTI-specific detail before submission.)*

1. **Preserve the legacy interface, adapt only what's behind it.** Wrapping
   the exact pre-existing XML-RPC contract (§4.4) meant the robot program —
   a validated, change-controlled artifact from the operator's point of
   view — never had to be touched or re-qualified to accept the adaptation
   loop. In an industrial setting, this reduced "what changed" to a single,
   reviewable surface.
2. **Separating policy from mechanism paid for itself at the second and
   third deployment tier, not the first.** The managed/managing split and
   keeping adaptation policy out of the managed system's configuration only
   *look* like architectural tidiness in simulation; the payoff was
   concrete once the same policy config had to drive three physically
   different adapters without divergence.
3. **Writing the trustworthiness specification is itself a bug-finding
   activity, not just a monitoring-deployment prerequisite.** Both latent
   defects in §6.4 were found while formalizing properties, before any
   checker ran against live data.
4. **In-loop and out-of-loop trust mechanisms catch different bug classes.**
   Legitimate can approve a locally-sound decision that is part of a
   globally pathological pattern; only a property-based, whole-protocol
   checker can see the pattern. Neither mechanism alone would have been
   "trustworthy adaptation" by itself.
5. **Cross-team contracts need to be written down explicitly, not assumed.**
   Splitting the specification (owned by this team) from the checker
   implementation (owned by a separate RV team) required an explicit,
   documented contract for tick semantics, sentinel values, and race-freedom
   assumptions; informal alignment would not have been sufficient once the
   two teams' schedules diverged.
6. **The graduated deployment strategy is a risk-reduction tool, not just a
   testing convenience** — §5.3's field incident was caught specifically
   because the light-box tier existed as an intermediate, lower-stakes place
   to re-validate a fix.

*(TODO — authors to expand: specific operator/technician feedback from DTI,
any process or governance changes DTI made as a result of adopting this
system, and concrete guidance for teams attempting a similar graduated
deployment in a different industrial setting.)*

---

## 8. Related Work

*(TODO: fill in citations — noting here the categories this section needs to
cover so the placeholders are actionable, not just "related work TBD.")*

- Self-adaptive systems and MAPE-K (Kephart & Chess' autonomic computing
  vision; the broader SEAMS community's reference architectures).
- The RoboSAPIENS project and its reference architecture for trustworthy
  self-adaptation in robotics.
- Runtime verification with stream-based specification languages (the LOLA
  family, and related tools for monitoring reactive/cyber-physical systems).
- Industrial experience reports on ML model management / model-switching
  under distribution shift in production.
- Robotic disassembly and vision-guided electronics refurbishment /
  circular-economy automation.

---

## 9. Threats to Validity / Limitations

- **Single site, single cell.** All field evidence comes from one robotic
  cell at one site (DTI); generalization to other refurbishment lines or
  other manufacturers' equipment is not established here.
- **Baseline log is from the pre-adaptive system.** The two-month,
  3,568-cycle characterization in §2.1 evidences the *problem's* frequency,
  not the *adaptive system's* effectiveness — see §6.3.
- **Light-box entropy is a fallback approximation**, not the exact field
  computation, because of a missing patched-YOLO dependency on that rig
  (§5.2); light-box trials validate protocol behavior, not exact field
  numbers.
- **The TC checker's two anticipated findings (§6.4) are not yet confirmed
  against live production traffic** — they are structurally certain given
  the current code, but we have not yet observed the monitor flag them in
  the field.
- **Specification language provenance.** The runtime-verification
  specification is LOLA-*inspired*, not a validated LOLA implementation;
  claims about its expressiveness or guarantees should not be read as
  transferring from the LOLA literature without qualification.

---

## 10. Conclusion

*(TODO: 1 paragraph, written last once §7 and §8 are finalized.)*

## Acknowledgments

This work was funded by the RoboSAPIENS project. We thank DTI for hosting
the deployment and providing production log access.

## References

*(TODO: BibTeX to be added — see §8 for the categories needed.)*
