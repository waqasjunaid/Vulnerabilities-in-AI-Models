"""Configuration for PromptArmor."""

from typing import List, Optional, Dict, Any
from dataclasses import dataclass, field
from enum import Enum


class LLMProvider(Enum):
    """Supported LLM providers."""

    ANTHROPIC = "anthropic"
    OPENAI = "openai"
    GEMINI = "gemini"


class EmbeddingProvider(Enum):
    """Supported embedding providers."""

    OPENAI = "openai"
    SENTENCE_TRANSFORMERS = "sentence_transformers"
    VERTEX = "vertex"


@dataclass
class CanaryConfig:
    """Configuration for canary token layer."""

    enabled: bool = True
    num_canaries: int = 3
    custom_canaries: List[str] = field(default_factory=list)
    auto_generate: bool = True
    alert_on_trigger: bool = True


@dataclass
class ClassifierConfig:
    """Configuration for attack classifier layer."""

    enabled: bool = True
    use_embedding_similarity: bool = True
    similarity_threshold: float = 0.85
    use_regex_patterns: bool = True
    custom_patterns: List[str] = field(default_factory=list)


@dataclass
class SanitizerConfig:
    """Configuration for input sanitizer layer."""

    enabled: bool = True
    normalize_unicode: bool = True
    decode_common_encodings: bool = True  # Base64, URL encoding, etc.
    remove_invisible_chars: bool = True
    max_input_length: int = 10000
    strip_html: bool = True


@dataclass
class JudgeConfig:
    """Configuration for LLM-as-Judge layer."""

    enabled: bool = True
    provider: LLMProvider = LLMProvider.ANTHROPIC
    model: str = "claude-3-haiku-20240307"
    temperature: float = 0.0
    threshold: float = 0.7  # Block if compromise probability > threshold
    timeout_seconds: float = 10.0


@dataclass
class DriftConfig:
    """Configuration for semantic drift detection layer."""

    enabled: bool = True
    embedding_provider: EmbeddingProvider = EmbeddingProvider.SENTENCE_TRANSFORMERS
    embedding_model: str = "all-MiniLM-L6-v2"
    drift_threshold: float = 0.6  # Flag if similarity < threshold
    expected_behaviors: List[str] = field(default_factory=list)


@dataclass
class SignatureConfig:
    """Configuration for response signature verification layer."""

    enabled: bool = True
    signature_prefix: str = "[ARMOR_SIG:"
    signature_suffix: str = "]"
    include_hash: bool = True
    hash_algorithm: str = "sha256"


@dataclass
class RedTeamConfig:
    """Configuration for red team simulation."""

    enabled: bool = True
    num_attacks_per_run: int = 50
    include_encoding_attacks: bool = True
    include_roleplay_attacks: bool = True
    include_multilingual_attacks: bool = True
    custom_attacks: List[str] = field(default_factory=list)
    save_successful_attacks: bool = True


@dataclass
class ArmorConfig:
    """Main configuration for PromptArmor."""

    # General settings
    name: str = "PromptArmor"
    strict_mode: bool = False  # Block on any detection vs. only high confidence
    log_all_requests: bool = True

    # LLM settings for main processing
    llm_provider: LLMProvider = LLMProvider.ANTHROPIC
    llm_model: str = "claude-3-sonnet-20240229"

    # System prompt for the protected LLM
    system_prompt: str = "You are a helpful assistant."

    # Layer configurations
    canary: CanaryConfig = field(default_factory=CanaryConfig)
    classifier: ClassifierConfig = field(default_factory=ClassifierConfig)
    sanitizer: SanitizerConfig = field(default_factory=SanitizerConfig)
    judge: JudgeConfig = field(default_factory=JudgeConfig)
    drift: DriftConfig = field(default_factory=DriftConfig)
    signature: SignatureConfig = field(default_factory=SignatureConfig)

    # Red team settings
    redteam: RedTeamConfig = field(default_factory=RedTeamConfig)

    # Layer execution order
    layer_order: List[str] = field(default_factory=lambda: [
        "sanitizer",
        "classifier",
        "canary",
        "drift",
        "judge",
        "signature",
    ])

    # Thresholds
    block_threshold: float = 0.7
    warn_threshold: float = 0.4

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            "name": self.name,
            "strict_mode": self.strict_mode,
            "llm_provider": self.llm_provider.value,
            "llm_model": self.llm_model,
            "layer_order": self.layer_order,
            "block_threshold": self.block_threshold,
            "warn_threshold": self.warn_threshold,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ArmorConfig":
        """Create config from dictionary."""
        config = cls()
        for key, value in data.items():
            if hasattr(config, key):
                setattr(config, key, value)
        return config

    @classmethod
    def strict(cls) -> "ArmorConfig":
        """Create a strict configuration that blocks aggressively."""
        return cls(
            strict_mode=True,
            block_threshold=0.5,
            warn_threshold=0.3,
        )

    @classmethod
    def permissive(cls) -> "ArmorConfig":
        """Create a permissive configuration for lower false positives."""
        return cls(
            strict_mode=False,
            block_threshold=0.85,
            warn_threshold=0.6,
        )
