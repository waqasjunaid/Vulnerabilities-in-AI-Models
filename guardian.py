#"""
#guardian.py
#-----------
#The defense layer (Step 2). A Guardian inspects each proposed tool call
#*before* it executes and decides whether to allow or block it.
#
#Guardians:
#  * NoGuardian      -- allows everything. The UNDEFENDED baseline
#                       (dh ASR 0.257, ds ASR 0.294).
#  * ProvenanceGate  -- non-learned floor: blocks any PRIVILEGED tool call once
#                       UNTRUSTED content has entered context. Near-zero ASR on
#                       InjecAgent, but 0% utility on the utility-stress set.
#  * DetectorGate    -- (in detectors.py) content-AWARE: scores untrusted
#                       content and blocks a privileged call only when the
#                       content looks like an injection. Recovers utility.
#
#The Guardian interface is the single hook every defense plugs into, so the
#agent code never changes again.
#"""
#
#from __future__ import annotations
#from dataclasses import dataclass
#
#
#@dataclass
#class Decision:
#    allow: bool
#    reason: str = ""
#    score: float = 0.0   # used by calibrated detectors (0 = clearly safe)
#
#
#class Guardian:
#    """Interface. inspect() is called before every tool execution."""
#
#    name = "base"
#
#    def inspect(self, action: str, action_input: dict,
#                trajectory: list[dict], env) -> Decision:
#        raise NotImplementedError
#
#
#class NoGuardian(Guardian):
#    """Allows everything -- the undefended baseline."""
#
#    name = "none"
#
#    def inspect(self, action, action_input, trajectory, env) -> Decision:
#        return Decision(allow=True, reason="no guardian")
#
#
#class ProvenanceGate(Guardian):
#    """
#    Non-learned floor defense. Blocks a privileged action if ANY untrusted
#    content has been ingested earlier in this trajectory.
#
#    strict : if True, block the privileged call outright. If False, only flag
#             it (allow=True but reason records the taint).
#    """
#
#    name = "provenance_gate"
#
#    def __init__(self, strict: bool = True):
#        self.strict = strict
#
#    def inspect(self, action, action_input, trajectory, env) -> Decision:
#        tool = env.tools.get(action)
#        is_privileged = bool(tool and tool.privileged)
#        if not is_privileged:
#            return Decision(allow=True, reason="non-privileged action")
#
#        seen_untrusted = any(step.get("trust") == "untrusted"
#                             for step in trajectory)
#        if not seen_untrusted:
#            return Decision(allow=True,
#                            reason="privileged action, no untrusted context")
#
#        reason = (f"privileged action '{action}' attempted after untrusted "
#                  f"content entered context")
#        if self.strict:
#            return Decision(allow=False, reason=reason, score=1.0)
#        return Decision(allow=True, reason="FLAGGED (non-strict): " + reason,
#                        score=1.0)
#
#
## Registry so the harness can select a guardian by name on the CLI.
## NOTE: every branch must come BEFORE the final raise, or it is unreachable.
#def make_guardian(name: str) -> Guardian:
#    if name in (None, "none"):
#        return NoGuardian()
#    if name == "provenance":
#        return ProvenanceGate(strict=True)
#    if name == "provenance-flag":
#        return ProvenanceGate(strict=False)
#    if name == "detector":                       # heuristic floor, no download
#        from detectors import DetectorGate, HeuristicDetector
#        return DetectorGate(HeuristicDetector(), threshold=0.35)
#    if name == "detector-deberta":               # learned baseline (offline model)
#        import os
#        from detectors import DetectorGate, DebertaDetector
#        path = os.environ.get("PI_DETECTOR_PATH", "deberta-pi")
##        return DetectorGate(DebertaDetector(path), threshold=0.5)
#        from deberta_detector import DebertaDetector          # the hardened one
#        return DetectorGate(DebertaDetector())                 # reads PI_DETECTOR_PATH; threshold via GUARDIAN_THRESHOLD
#    raise SystemExit(f"unknown guardian: {name}")
#
#
## Single source of truth for the CLI choices. Import this in the runners so
## argparse and make_guardian can never drift apart again.
#GUARDIAN_CHOICES = [
#    "none", "provenance", "provenance-flag", "detector", "detector-deberta",
#]



"""
guardian.py
-----------
The defense layer (Step 2). A Guardian inspects each proposed tool call
*before* it executes and decides whether to allow or block it.

Guardians:
  * NoGuardian      -- allows everything. The UNDEFENDED baseline
                       (dh ASR 0.257, ds ASR 0.294).
  * ProvenanceGate  -- non-learned floor: blocks any PRIVILEGED tool call once
                       UNTRUSTED content has entered context. Near-zero ASR on
                       InjecAgent, but 0% utility on the utility-stress set.
  * DetectorGate    -- (in detectors.py) content-AWARE: scores untrusted
                       content and blocks a privileged call only when the
                       content looks like an injection. Recovers utility.
  * TrustCalibratedGate -- contextual action-boundary defense. Combines the
                           detector evidence with provenance, action sensitivity,
                           and trajectory/action alignment.

The Guardian interface is the single hook every defense plugs into, so the
agent code never changes again.
"""

from __future__ import annotations
from dataclasses import dataclass
import os


@dataclass
class Decision:
    allow: bool
    reason: str = ""
    score: float = 0.0   # used by calibrated detectors (0 = clearly safe)


class Guardian:
    """Interface. inspect() is called before every tool execution."""

    name = "base"

    def inspect(self, action: str, action_input: dict,
                trajectory: list[dict], env) -> Decision:
        raise NotImplementedError


class NoGuardian(Guardian):
    """Allows everything -- the undefended baseline."""

    name = "none"

    def inspect(self, action, action_input, trajectory, env) -> Decision:
        return Decision(allow=True, reason="no guardian")


class ProvenanceGate(Guardian):
    """
    Non-learned floor defense. Blocks a privileged action if ANY untrusted
    content has been ingested earlier in this trajectory.

    strict : if True, block the privileged call outright. If False, only flag
             it (allow=True but reason records the taint).
    """

    name = "provenance_gate"

    def __init__(self, strict: bool = True):
        self.strict = strict

    def inspect(self, action, action_input, trajectory, env) -> Decision:
        tool = env.tools.get(action)
        is_privileged = bool(tool and tool.privileged)
        if not is_privileged:
            return Decision(allow=True, reason="non-privileged action")

        seen_untrusted = any(step.get("trust") == "untrusted"
                             for step in trajectory)
        if not seen_untrusted:
            return Decision(allow=True,
                            reason="privileged action, no untrusted context")

        reason = (f"privileged action '{action}' attempted after untrusted "
                  f"content entered context")
        if self.strict:
            return Decision(allow=False, reason=reason, score=1.0)
        return Decision(allow=True, reason="FLAGGED (non-strict): " + reason,
                        score=1.0)


# Registry so the harness can select a guardian by name on the CLI.
# NOTE: every branch must come BEFORE the final raise, or it is unreachable.
def make_guardian(name: str) -> Guardian:
    if name in (None, "none"):
        return NoGuardian()
    if name == "provenance":
        return ProvenanceGate(strict=True)
    if name == "provenance-flag":
        return ProvenanceGate(strict=False)
    if name == "detector":                       # heuristic floor, no download
        from detectors import DetectorGate, HeuristicDetector
        return DetectorGate(HeuristicDetector(), threshold=0.35)
    if name == "detector-deberta":               # learned baseline (offline model)
        from detectors import DetectorGate, DebertaDetector
        path = os.environ.get("PI_DETECTOR_PATH", "deberta-pi")
#        return DetectorGate(DebertaDetector(path), threshold=0.5)
        from deberta_detector import DebertaDetector          # the hardened one
        return DetectorGate(DebertaDetector())                 # reads PI_DETECTOR_PATH; threshold via GUARDIAN_THRESHOLD
    if name == "trust-calibrated":            # contextual trust guardian v0
        from detectors import HeuristicDetector
        from trust_model import TrustCalibratedGuardian
        detector_kind = os.environ.get("TRUST_DETECTOR", "heuristic").strip().lower()
        if detector_kind == "heuristic":
            detector = HeuristicDetector()
        elif detector_kind == "deberta":
            from deberta_detector import DebertaDetector
            detector = DebertaDetector()
        else:
            raise SystemExit(
                "unknown TRUST_DETECTOR: %s (use heuristic or deberta)"
                % detector_kind
            )
        return TrustCalibratedGuardian(detector=detector)
    raise SystemExit(f"unknown guardian: {name}")


# Single source of truth for the CLI choices. Import this in the runners so
# argparse and make_guardian can never drift apart again.
GUARDIAN_CHOICES = [
    "none", "provenance", "provenance-flag", "detector", "detector-deberta",
    "trust-calibrated",
]