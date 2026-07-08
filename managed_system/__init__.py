"""MANAGED SYSTEM - the screw-detection application that does the work.

Internal layering (unchanged from before the managed/managing split):

    core.py       hardware-agnostic primitives (ScrewDetectionCore) and the
                  probe/effector surface the managing system calls
    adapters/     the hardware behind the core: sim | real | lightbox

The managing system (MAPLE-K loop, `managing_system/`) touches this package
ONLY through the probe/effector methods on `ScrewDetectionCore` - see the
interface notes in core.py. Nothing in here knows that MAPLE-K exists.
"""
