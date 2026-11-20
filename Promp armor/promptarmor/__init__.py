"""
PromptArmor - Multi-Layer Prompt Injection Defense System

A comprehensive defense system that treats prompt injection like an immune system:
multiple detection mechanisms, learns from attacks, and includes red team simulation.
"""

from promptarmor.core.armor import PromptArmor
from promptarmor.core.config import ArmorConfig
from promptarmor.core.types import (
    ThreatLevel,
    DetectionResult,
    LayerResult,
    ArmorResponse,
    AttackPattern,
)
from promptarmor.layers import (
    CanaryLayer,
    ClassifierLayer,
    SanitizerLayer,
    JudgeLayer,
    DriftLayer,
    SignatureLayer,
)

__version__ = "0.1.0"
__author__ = "Francisco Perez"

__all__ = [
    "PromptArmor",
    "ArmorConfig",
    "ThreatLevel",
    "DetectionResult",
    "LayerResult",
    "ArmorResponse",
    "AttackPattern",
    "CanaryLayer",
    "ClassifierLayer",
    "SanitizerLayer",
    "JudgeLayer",
    "DriftLayer",
    "SignatureLayer",
]
