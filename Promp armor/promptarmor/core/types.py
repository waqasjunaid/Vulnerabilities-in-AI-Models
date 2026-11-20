"""Type definitions for PromptArmor."""

from enum import Enum
from typing import Optional, List, Dict, Any
from dataclasses import dataclass, field
from datetime import datetime


class ThreatLevel(Enum):
    """Threat level classification for detected attacks."""

    NONE = 0        # No threat detected
    LOW = 1         # Suspicious but likely benign
    MEDIUM = 2      # Possible injection attempt
    HIGH = 3        # Likely injection attack
    CRITICAL = 4    # Confirmed injection / canary triggered


class AttackCategory(Enum):
    """Categories of prompt injection attacks."""

    DIRECT_OVERRIDE = "direct_override"         # "Ignore previous instructions"
    ROLE_PLAY = "role_play"                     # "You are now DAN..."
    CONTEXT_MANIPULATION = "context_manipulation"  # Fake system messages
    ENCODING_BYPASS = "encoding_bypass"         # Base64, ROT13, etc.
    MULTI_TURN = "multi_turn"                   # Gradual manipulation
    PAYLOAD_SPLITTING = "payload_splitting"     # Split malicious content
    VIRTUALIZATION = "virtualization"           # "Pretend you're in a sandbox"
    EMOTIONAL_MANIPULATION = "emotional_manipulation"  # "My grandma used to..."
    AUTHORITY_IMPERSONATION = "authority_impersonation"  # "As an OpenAI engineer..."
    INSTRUCTION_INJECTION = "instruction_injection"  # Hidden in user data


@dataclass
class AttackPattern:
    """Represents a known attack pattern."""

    id: str
    name: str
    category: AttackCategory
    pattern: str  # Regex or exact match
    description: str
    severity: ThreatLevel
    examples: List[str] = field(default_factory=list)
    bypass_techniques: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "category": self.category.value,
            "pattern": self.pattern,
            "description": self.description,
            "severity": self.severity.value,
            "examples": self.examples,
            "bypass_techniques": self.bypass_techniques,
        }


@dataclass
class LayerResult:
    """Result from a single defense layer."""

    layer_name: str
    passed: bool
    threat_level: ThreatLevel
    confidence: float  # 0.0 to 1.0
    details: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    processing_time_ms: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "layer_name": self.layer_name,
            "passed": self.passed,
            "threat_level": self.threat_level.value,
            "confidence": self.confidence,
            "details": self.details,
            "metadata": self.metadata,
            "processing_time_ms": self.processing_time_ms,
        }


@dataclass
class DetectionResult:
    """Aggregated result from all defense layers."""

    is_safe: bool
    overall_threat_level: ThreatLevel
    blocked: bool
    block_reason: Optional[str]
    layer_results: List[LayerResult]
    total_processing_time_ms: float
    timestamp: datetime = field(default_factory=datetime.utcnow)

    @property
    def threat_score(self) -> float:
        """Calculate overall threat score (0.0 to 1.0)."""
        if not self.layer_results:
            return 0.0

        # Weighted average based on confidence
        total_weight = 0.0
        weighted_sum = 0.0

        for result in self.layer_results:
            weight = result.confidence
            score = result.threat_level.value / ThreatLevel.CRITICAL.value
            weighted_sum += weight * score
            total_weight += weight

        return weighted_sum / total_weight if total_weight > 0 else 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "is_safe": self.is_safe,
            "overall_threat_level": self.overall_threat_level.value,
            "threat_score": self.threat_score,
            "blocked": self.blocked,
            "block_reason": self.block_reason,
            "layer_results": [r.to_dict() for r in self.layer_results],
            "total_processing_time_ms": self.total_processing_time_ms,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class ArmorResponse:
    """Complete response from PromptArmor processing."""

    original_input: str
    sanitized_input: Optional[str]
    llm_response: Optional[str]
    final_response: Optional[str]
    detection_result: DetectionResult
    canaries_triggered: List[str] = field(default_factory=list)
    signature_valid: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "original_input": self.original_input,
            "sanitized_input": self.sanitized_input,
            "llm_response": self.llm_response,
            "final_response": self.final_response,
            "detection_result": self.detection_result.to_dict(),
            "canaries_triggered": self.canaries_triggered,
            "signature_valid": self.signature_valid,
        }


@dataclass
class CanaryToken:
    """A hidden canary token for detecting prompt leaks."""

    id: str
    token: str
    threat_level: ThreatLevel
    description: str
    triggered_count: int = 0
    last_triggered: Optional[datetime] = None

    def trigger(self) -> None:
        """Record a canary trigger event."""
        self.triggered_count += 1
        self.last_triggered = datetime.utcnow()


@dataclass
class RedTeamResult:
    """Result from a red team attack simulation."""

    attack_id: str
    attack_name: str
    attack_payload: str
    succeeded: bool
    response: str
    detection_result: DetectionResult
    bypass_method: Optional[str] = None
    notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "attack_id": self.attack_id,
            "attack_name": self.attack_name,
            "attack_payload": self.attack_payload,
            "succeeded": self.succeeded,
            "response": self.response,
            "detection_result": self.detection_result.to_dict(),
            "bypass_method": self.bypass_method,
            "notes": self.notes,
        }


@dataclass
class GameSession:
    """Escape room game session state."""

    session_id: str
    player_name: str
    attempts: int = 0
    max_attempts: int = 10
    successful_attacks: List[str] = field(default_factory=list)
    failed_attacks: List[str] = field(default_factory=list)
    hints_used: int = 0
    score: int = 0
    started_at: datetime = field(default_factory=datetime.utcnow)
    completed: bool = False

    @property
    def attempts_remaining(self) -> int:
        return self.max_attempts - self.attempts

    def record_attempt(self, payload: str, succeeded: bool) -> None:
        self.attempts += 1
        if succeeded:
            self.successful_attacks.append(payload)
            self.score += 100 - (self.hints_used * 10)
        else:
            self.failed_attacks.append(payload)
