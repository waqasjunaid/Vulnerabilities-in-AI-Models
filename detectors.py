"""
detectors.py
------------
Stage 2 of the guardian: the CONTENT-AWARE detector layer.

Why this exists
---------------
ProvenanceGate (Stage 1) is content-blind: it blocks a privileged tool call
whenever ANY untrusted content is in context. That gives near-zero ASR on
InjecAgent but 0% utility on the utility-stress set, because it cannot tell a
benign untrusted observation from a malicious one. This layer closes exactly
that gap: it SCORES the untrusted content and blocks the privileged call only
when the content actually looks like an injection.

The gate stays identical to ProvenanceGate in every other respect, so your
2x2 headline experiment (ds/attacks x stress/legit, gate vs detector) drops
straight out of the same runner and the same metrics.

Detectors provided
-------------------
  * HeuristicDetector      -- pure Python, no model, runs offline immediately.
                              The FLOOR. Its numbers are a plumbing/mechanism
                              check, NOT a result for the paper.
  * DebertaDetector        -- wraps the ProtectAI deberta-v3 prompt-injection
                              classifier. The cheap learned baseline to beat.
                              Transfer the model offline like you did Qwen.
  * TrustCalibratedDetector-- STUB. This is YOUR contribution slot: a detector
                              trained with a Train-Trust-style trust-calibrated
                              objective. Interface is wired; model comes later.

All detectors implement the same contract:
    score(text: str) -> float   # in [0, 1]; higher = more likely an injection
so the gate and the threshold sweep never care which one is loaded.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass

# Reuse the exact interface the agent already calls into.
from guardian import Decision, Guardian


# --------------------------------------------------------------------------- #
# Detector interface
# --------------------------------------------------------------------------- #
class Detector:
    """A detector scores a single piece of untrusted content for injection."""
    name = "base-detector"

    def score(self, text: str) -> float:
        """Return an injection likelihood in [0, 1]. 0 = clearly benign."""
        raise NotImplementedError


# --------------------------------------------------------------------------- #
# 1. HeuristicDetector -- offline floor, no dependencies
# --------------------------------------------------------------------------- #
class HeuristicDetector(Detector):
    """
    A transparent, dependency-free floor. It looks for the fingerprints of an
    indirect prompt injection embedded inside data the agent merely READ:

      (a) instruction-override framing  ("ignore previous", "you are now", ...)
      (b) imperative verbs that map to PRIVILEGED actions (transfer, send,
          delete, forward, wire, pay, grant, ...)
      (c) role / turn spoofing          ("system:", "assistant:", ...)
      (d) exfil targets embedded in prose (urls, emails, accounts)

    Each group contributes weight; the score saturates at 1.0. This is NOT a
    trained model and will both miss paraphrased attacks and occasionally
    over-fire -- which is the whole point of calling it a floor. Report it as
    the floor your DebertaDetector, then your TrustCalibratedDetector, beat.

    Empirically, on InjecAgent ds-base content this tops out at 0.45
    (ACTION_VERB 0.20 + EXFIL 0.25), while the benign utility-stress content
    scores <= 0.20. That 0.20 / 0.45 gap is why the gate threshold must sit
    below 0.45; see DetectorGate below.
    """
    name = "heuristic"

    OVERRIDE = re.compile(
        r"\b(ignore|disregard|forget|override)\b.{0,30}"
        r"\b(previous|prior|above|earlier|all)\b"
        r"|(you are now|new instructions?|instead of|before (you |)"
        r"(summariz|repl|answer|respond|continu))",
        re.I | re.S,
    )
    ACTION_VERB = re.compile(
        r"\b(transfer|send|forward|delete|remove|wire|pay|purchase|buy|book|"
        r"grant|revoke|share|post|publish|email|message|export|upload|"
        r"execute|run|install|disable|approve)\b",
        re.I,
    )
    TURN_SPOOF = re.compile(r"(^|\n)\s*(system|assistant|user)\s*:", re.I)
    EXFIL = re.compile(
        r"(https?://\S+|\b[\w.+-]+@[\w-]+\.[\w.-]+\b|\b(?:account|iban|api[_-]?key|"
        r"token|password|secret)\b)",
        re.I,
    )
    IMPERATIVE_LEAD = re.compile(
        r"(^|\n|\.)\s*(please\s+)?(immediately\s+)?"
        r"(transfer|send|forward|delete|wire|pay|grant|revoke|share|post|email|"
        r"export|upload|execute|run|install|disable|approve)\b",
        re.I,
    )

    def score(self, text: str) -> float:
        if not text:
            return 0.0
        t = text.strip()
        s = 0.0
        if self.OVERRIDE.search(t):
            s += 0.55                         # override framing is a strong tell
        if self.TURN_SPOOF.search(t):
            s += 0.45                         # data pretending to be a chat turn
        if self.IMPERATIVE_LEAD.search(t):
            s += 0.40                         # a command leading a sentence
        elif self.ACTION_VERB.search(t):
            s += 0.20                         # weaker: verb present but not leading
        if self.EXFIL.search(t) and self.ACTION_VERB.search(t):
            s += 0.25                         # target + action = exfil pattern
        return min(1.0, s)


# --------------------------------------------------------------------------- #
# 2. DebertaDetector -- the learned baseline (offline model)
# --------------------------------------------------------------------------- #
class DebertaDetector(Detector):
    """
    Wraps ProtectAI's deberta-v3 prompt-injection classifier. Transfer the
    model to the offline box the same way you did Qwen:

      # on a connected machine
      huggingface-cli download ProtectAI/deberta-v3-base-prompt-injection-v2 \\
          --local-dir deberta-pi
      # copy the 'deberta-pi' folder to the ThinkStation, then:
      export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

    Pass the local path (or set PI_DETECTOR_PATH). The label meaning INJECTION
    is auto-detected from the model config, so you don't hard-code label order.
    Long observations are chunked; the score is the max over chunks (an
    injection anywhere in the content is enough to flag it).
    """
    name = "deberta"

    def __init__(self, model_path: str, max_chars: int = 1500):
        from transformers import (AutoModelForSequenceClassification,
                                  AutoTokenizer)
        import torch
        self.torch = torch
        self.tok = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.model.eval()
        self.max_chars = max_chars
        # Find which class index means "injection" from id2label.
        id2label = self.model.config.id2label
        self._inj_idx = 0
        for idx, lab in id2label.items():
            if re.search(r"inj|attack|unsafe|1", str(lab), re.I):
                self._inj_idx = int(idx)
                break

    def _chunks(self, text: str):
        for i in range(0, len(text), self.max_chars):
            yield text[i:i + self.max_chars]

    def score(self, text: str) -> float:
        if not text or not text.strip():
            return 0.0
        best = 0.0
        with self.torch.no_grad():
            for chunk in self._chunks(text):
                enc = self.tok(chunk, return_tensors="pt", truncation=True,
                               max_length=512)
                logits = self.model(**enc).logits
                prob = self.torch.softmax(logits, dim=-1)[0, self._inj_idx].item()
                best = max(best, prob)
        return best


# --------------------------------------------------------------------------- #
# 3. TrustCalibratedDetector -- YOUR CONTRIBUTION (stub)
# --------------------------------------------------------------------------- #
class TrustCalibratedDetector(Detector):
    """
    STUB / interface placeholder for the thesis contribution.

    This is where the Train-Trust transfer lands: a detector whose objective is
    an explicit trust-calibrated, robustness-aware loss (not plain BCE), so the
    score it returns is a *calibrated* trust signal you can sweep to trace the
    safety-utility curve, and so it holds up under adaptive attacks (Stage 4).

    Until it's trained it delegates to a fallback detector so the pipeline runs
    end to end. Replace `score` with a forward pass through your trained model.
    """
    name = "trust-calibrated"

    def __init__(self, fallback: Detector | None = None):
        self.fallback = fallback or HeuristicDetector()

    def score(self, text: str) -> float:
        # TODO(thesis): load the trust-calibrated checkpoint and return its
        # calibrated injection/trust score here.
        return self.fallback.score(text)


# --------------------------------------------------------------------------- #
# The gate: content-aware, drop-in replacement for ProvenanceGate
# --------------------------------------------------------------------------- #
class DetectorGate(Guardian):
    """
    Same shape as ProvenanceGate, but content-AWARE.

    Logic for a proposed tool call:
      1. Not privileged            -> allow.
      2. Privileged, no untrusted  -> allow (nothing untrusted could drive it).
      3. Privileged + untrusted    -> score every untrusted observation seen so
                                      far; block iff max score >= threshold.

    Step 3 is the whole difference from the blunt gate: benign untrusted
    content scores low and the legitimate privileged action goes through
    (utility recovered on the stress set), while injected untrusted content
    scores high and is blocked (ASR stays low on ds/dh).

    Threshold / operating point
    ---------------------------
    `threshold` is your operating point. Sweeping it from 0->1 traces the
    safety-utility curve reviewers ask for; a single number hides that curve.

    With the HeuristicDetector the usable band is narrow: attacks score 0.45,
    benign scores <= 0.20, so the threshold must land in (0.20, 0.45]. The
    default here is 0.35 (mid-gap). To sweep WITHOUT editing code, set the
    GUARDIAN_THRESHOLD env var per run, e.g.

        GUARDIAN_THRESHOLD=0.30 python load_injecagent.py ... --guardian detector

    An explicit `threshold=` argument always wins over the env var, which wins
    over the 0.35 default. NOTE: make_guardian() must construct the gate
    WITHOUT hard-coding threshold=0.5, or that 0.5 overrides everything here
    and (being above 0.45) blocks nothing.
    """
    name = "detector_gate"

    DEFAULT_THRESHOLD = 0.35

    def __init__(self, detector: Detector, threshold: float | None = None):
        self.detector = detector
        if threshold is None:
            env = os.environ.get("GUARDIAN_THRESHOLD")
            threshold = float(env) if env is not None else self.DEFAULT_THRESHOLD
        self.threshold = threshold

    # The LIVE trajectory step uses 'observation'. The DUMPED record (jsonl)
    # renames it to 'obs'. Read both so the gate works against either shape.
    # Getting this wrong silently turns the gate into a no-op: every score is
    # 0.0, nothing is ever blocked, and ASR equals the undefended baseline.
    OBS_KEYS = ("observation", "obs")

    @staticmethod
    def _obs_text(step: dict) -> str:
        for k in DetectorGate.OBS_KEYS:
            v = step.get(k)
            if v:
                return str(v)
        return ""

    def inspect(self, action, action_input, trajectory, env) -> Decision:
        tool = env.tools.get(action)
        is_privileged = bool(tool and tool.privileged)
        if not is_privileged:
            return Decision(allow=True, reason="non-privileged action")

        untrusted_steps = [s for s in trajectory if s.get("trust") == "untrusted"]
        if not untrusted_steps:
            return Decision(allow=True,
                            reason="privileged action, no untrusted context")

        untrusted_obs = [self._obs_text(s) for s in untrusted_steps]

        # Fail LOUD, not silently-open. If untrusted steps exist but carry no
        # readable text, the key name has drifted again and the gate would
        # degrade into "allow everything". Refuse to pretend that is a defense.
        if not any(o.strip() for o in untrusted_obs):
            raise RuntimeError(
                f"DetectorGate: {len(untrusted_steps)} untrusted step(s) but no "
                f"observation text found under keys {self.OBS_KEYS}. "
                f"Step keys present: {sorted(untrusted_steps[0].keys())}. "
                f"Fix OBS_KEYS -- otherwise this gate silently allows everything."
            )

        max_score = max(self.detector.score(o) for o in untrusted_obs)
        if max_score >= self.threshold:
            return Decision(
                allow=False, score=max_score,
                reason=(f"privileged action '{action}' blocked: untrusted "
                        f"content scored {max_score:.2f} >= {self.threshold:.2f} "
                        f"({self.detector.name})"))
        return Decision(
            allow=True, score=max_score,
            reason=(f"privileged action '{action}' allowed: untrusted content "
                    f"scored {max_score:.2f} < {self.threshold:.2f} "
                    f"({self.detector.name})"))