"""Argument-grounded trust guardian (v2).

v1 (`TrustCalibratedGuardian`) scores the risk of a privileged action mostly
from how injection-like the untrusted text seen *anywhere* in the trajectory
is.  Once such text has been read, every later privileged action is penalised,
including actions that match the user's request exactly, and actions that match
the user's operation but carry attacker-chosen arguments can still pass.

v2 adds one information-flow check on top of v1 and leaves v1 untouched:

* An *injection span* is the part of an untrusted observation that starts at an
  explicit instruction marker ("IMPORTANT!!!", "ignore previous instructions",
  ...).  An observation the detector flags at or above the guardian threshold
  without any marker is treated as a span as a whole.
* An argument *entity* is a string/number leaf of the tool call (plus the
  emails, URLs and IBAN-like ids found inside longer strings).
* An entity is *attacker-authored* when it occurs in an injection span and not
  in the user's own request.  A call with any attacker-authored entity, or a
  long argument that repeats injected text verbatim, is blocked.  This applies
  to every tool, not only privileged ones, because reading an attacker URL is
  already an effect.
* A leaf that itself contains an instruction marker is a verbatim reference to a
  poisoned resource (for example a channel whose name is the injected text), not
  an attacker-chosen parameter; its content is not checked.
* Read-only tools are checked for destination-like values only (emails, URLs,
  IBANs): naming an existing resource is not an attacker-controlled sink.
* If v1 would block a privileged call, but the call is fully goal-aligned and
  no attacker-authored value reached its arguments, v2 allows it, provided
  every suspicious observation has an identifiable span and at least one span
  was located (suspicion that cannot be located keeps the v1 decision).

No task names, URLs or tool arguments of any benchmark appear here.
"""
from __future__ import annotations

import re
from typing import Iterable

MARKER_RE = re.compile(
    r"IMPORTANT\s*!*"
    r"|ignore\s+(?:all\s+)?(?:previous|prior|earlier)\s+instructions?"
    r"|you are now|new instructions?"
    r"|(?:^|\n)\s*(?:system|assistant)\s*:",
    re.I,
)

ECHO_WORDS = 8
MIN_ENTITY_LEN = 3
MAX_LEAF_ENTITY_LEN = 80

_CONTENT_KEYS = {"body", "content", "subject", "message", "text", "description"}
_EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")
_URL_RE = re.compile(
    r"(?<![@\w.-])(?:https?://)?(?:www\.)?[\w-]+(?:\.[\w-]+)*\.[a-z]{2,}(?:/[^\s,;)\"']*)?(?![@\w-])",
    re.I,
)
_IBAN_RE = re.compile(r"\b[A-Z]{2}\d{2}[A-Z0-9]{8,30}\b")
_NUMERIC_RE = re.compile(r"[\d.,]+")
_WORD_RE = re.compile(r"[a-z0-9']+")


def content_fields(text: str) -> list[str]:
    """Semantic text fields of an observation, or the whole text if unstructured."""
    try:
        import yaml

        parsed = yaml.safe_load(text)
    except Exception:
        return [text]
    fields: list[str] = []

    def visit(value):
        if isinstance(value, dict):
            for key, child in value.items():
                if str(key).lower() in _CONTENT_KEYS and isinstance(child, str):
                    fields.append(child)
                else:
                    visit(child)
        elif isinstance(value, (list, tuple)):
            for child in value:
                visit(child)

    visit(parsed)
    return fields or [text]


def injection_spans(text: str, detector, strong: float) -> tuple[list[str], bool]:
    """Return (spans, unexplained) for one untrusted observation.

    `unexplained` is True when the detector flags the observation at or above
    `strong` but no explicit marker locates the injected text.
    """
    fields = content_fields(text)
    spans = []
    for field in fields:
        m = MARKER_RE.search(field)
        if m:
            spans.append(field[m.start():])
    if spans:
        return spans, False
    joined = " ".join(fields)
    try:
        score = float(detector.score(joined)) if joined.strip() else 0.0
    except Exception:
        score = 0.0
    if score >= strong:
        return [joined], True
    return [], False


def _leaves(value) -> Iterable[str]:
    if isinstance(value, dict):
        for child in value.values():
            yield from _leaves(child)
    elif isinstance(value, (list, tuple, set)):
        for child in value:
            yield from _leaves(child)
    elif value is None or isinstance(value, bool):
        return
    elif isinstance(value, (int, float)):
        yield format(value, "g")
    elif isinstance(value, str):
        yield value
    else:
        yield str(value)


def _norm(entity: str) -> str:
    e = entity.strip().strip("'\"`.,;:()[]{}").lower()
    e = re.sub(r"^https?://", "", e)
    e = re.sub(r"\s+", " ", e).rstrip("/")
    return e


def argument_entities(action_input) -> tuple[set[str], list[str]]:
    """Entities of a call, plus the long free-text arguments (for echo checks)."""
    entities: set[str] = set()
    long_texts: list[str] = []
    for leaf in _leaves(action_input):
        s = leaf.strip()
        if not s or MARKER_RE.search(s):
            continue
        if len(s) <= MAX_LEAF_ENTITY_LEN:
            entities.add(s)
        else:
            long_texts.append(s)
        for rx in (_EMAIL_RE, _URL_RE, _IBAN_RE):
            entities.update(m.group(0) for m in rx.finditer(s))
    normalised = set(map(_norm, entities))
    normalised.update(re.split(r"[/?#]", e, maxsplit=1)[0] for e in list(normalised) if "." in e)
    return {e for e in normalised if len(e) >= MIN_ENTITY_LEN or (e.isdigit() and len(e) >= 2)}, long_texts


def contains(haystack_lower: str, entity: str) -> bool:
    pattern = r"(?<![a-z0-9])" + re.escape(entity) + r"(?![a-z0-9])"
    if _NUMERIC_RE.fullmatch(entity):
        pattern = r"(?<![a-z0-9])(?<!\d\.)" + re.escape(entity) + r"(?![a-z0-9])(?!\.\d)"
    return re.search(pattern, haystack_lower) is not None


def _shingles(text: str, n: int) -> set[tuple[str, ...]]:
    words = _WORD_RE.findall(text.lower())
    return {tuple(words[i:i + n]) for i in range(len(words) - n + 1)}


def is_destination(entity: str) -> bool:
    """Email address, URL/domain or IBAN-like id: somewhere data can be sent."""
    return (
        _EMAIL_RE.fullmatch(entity) is not None
        or _URL_RE.fullmatch(entity) is not None
        or re.fullmatch(_IBAN_RE.pattern, entity, re.I) is not None
    )


def tainted_entities(action_input, goal: str, spans: list[str], destinations_only: bool = False) -> list[str]:
    """Attacker-authored values in the arguments of one call."""
    if not spans:
        return []
    goal_l = (goal or "").lower()
    spans_l = [s.lower() for s in spans]
    entities, long_texts = argument_entities(action_input)
    if destinations_only:
        entities = {e for e in entities if is_destination(e)}
        long_texts = []
    found = sorted(
        e for e in entities
        if not contains(goal_l, e) and any(contains(sp, e) for sp in spans_l)
    )
    if long_texts:
        goal_sh = _shingles(goal_l, ECHO_WORDS)
        span_sh = set().union(*(_shingles(sp, ECHO_WORDS) for sp in spans_l))
        for text in long_texts:
            if (_shingles(text, ECHO_WORDS) & span_sh) - goal_sh:
                found.append("<verbatim copy of injected text>")
                break
    return found


def collect_spans(trajectory: list[dict], obs_text, detector, strong: float) -> tuple[list[str], bool]:
    spans: list[str] = []
    unexplained = False
    for step in trajectory:
        if step.get("trust") != "untrusted":
            continue
        text = obs_text(step)
        if not text or not text.strip():
            continue
        found, flag = injection_spans(text, detector, strong)
        spans.extend(found)
        unexplained = unexplained or flag
    return spans, unexplained


def _build_guardian_class():
    from guardian import Decision
    from trust_model import TrustCalibratedGuardian

    class ArgumentGroundingGuardian(TrustCalibratedGuardian):
        name = "argument_grounding_v2"
        RELIEF_ALIGNMENT = 0.999

        def inspect(self, action, action_input, trajectory, env):
            # A failed or empty tool result carries no external text; v1 raises
            # when every untrusted step is empty, so drop those steps here.
            trajectory = [
                s for s in trajectory
                if not (s.get("trust") == "untrusted"
                        and not self.evidence_model._obs_text(s).strip())
            ]
            strong = self.calibrator.threshold
            spans, unexplained = collect_spans(
                trajectory, self.evidence_model._obs_text, self.detector, strong
            )
            if spans:
                tool = (getattr(env, "tools", None) or {}).get(action)
                tainted = tainted_entities(
                    action_input, self.current_user_goal, spans,
                    destinations_only=not bool(tool and getattr(tool, "privileged", False)),
                )
                if tainted:
                    self.last_decision = None
                    shown = ", ".join(repr(t[:40]) for t in tainted[:3])
                    return Decision(
                        allow=False,
                        score=1.0,
                        reason=(
                            f"blocked by argument grounding: attacker-authored value(s) "
                            f"{shown} reach the arguments of '{action}'; "
                            f"tainted={len(tainted)}"
                        ),
                    )

            decision = super().inspect(action, action_input, trajectory, env)
            if decision.allow or unexplained or not spans:
                return decision

            last = self.last_decision
            if (
                last is not None
                and last.evidence.goal_compatibility_known
                and last.evidence.goal_alignment >= self.RELIEF_ALIGNMENT
            ):
                return Decision(
                    allow=True,
                    score=decision.score,
                    reason=(
                        f"privileged action '{action}' allowed by argument grounding: "
                        "fully goal-aligned and no attacker-authored value in its "
                        f"arguments; tainted=0; v1: {decision.reason}"
                    ),
                )
            return decision

    return ArgumentGroundingGuardian


def __getattr__(name):
    if name == "ArgumentGroundingGuardian":
        cls = _build_guardian_class()
        globals()[name] = cls
        return cls
    raise AttributeError(name)
