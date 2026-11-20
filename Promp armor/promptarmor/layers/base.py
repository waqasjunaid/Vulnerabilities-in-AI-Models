"""Base class for all defense layers."""

from abc import ABC, abstractmethod
from typing import Optional, Dict, Any
import time

from promptarmor.core.types import LayerResult, ThreatLevel


class BaseLayer(ABC):
    """
    Abstract base class for all defense layers.

    Each layer implements a specific detection mechanism and returns
    a LayerResult indicating whether the input/output passed the check.

    Layers can be:
    - Pre-LLM: Analyze user input before sending to LLM
    - Post-LLM: Analyze LLM response after generation

    Example implementation:
        ```python
        class MyCustomLayer(BaseLayer):
            layer_name = "my_custom_layer"

            async def _analyze_impl(
                self,
                content: str,
                context: Dict[str, Any]
            ) -> LayerResult:
                # Your detection logic here
                is_suspicious = "hack" in content.lower()

                return LayerResult(
                    layer_name=self.layer_name,
                    passed=not is_suspicious,
                    threat_level=ThreatLevel.HIGH if is_suspicious else ThreatLevel.NONE,
                    confidence=0.9,
                    details="Detected suspicious keyword" if is_suspicious else "Clean",
                )
        ```
    """

    layer_name: str = "base"

    def __init__(self, config: Any = None, armor_config: Any = None):
        """
        Initialize the layer.

        Args:
            config: Layer-specific configuration
            armor_config: Global ArmorConfig for shared settings
        """
        self.config = config
        self.armor_config = armor_config
        self._call_count = 0
        self._total_time_ms = 0.0

    @classmethod
    async def create(
        cls,
        config: Any = None,
        armor_config: Any = None,
    ) -> "BaseLayer":
        """
        Factory method to create and initialize a layer.

        Override this method if your layer needs async initialization.
        """
        instance = cls(config=config, armor_config=armor_config)
        await instance._initialize()
        return instance

    async def _initialize(self) -> None:
        """
        Perform async initialization.

        Override this method for layers that need to load models,
        connect to services, etc.
        """
        pass

    async def analyze(
        self,
        content: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> LayerResult:
        """
        Analyze content and return detection result.

        This method handles timing and statistics. Override _analyze_impl
        for custom detection logic.

        Args:
            content: The content to analyze (user input or LLM response)
            context: Optional context with additional information

        Returns:
            LayerResult with detection findings
        """
        start_time = time.perf_counter()
        context = context or {}

        try:
            result = await self._analyze_impl(content, context)
        except Exception as e:
            result = LayerResult(
                layer_name=self.layer_name,
                passed=True,  # Fail open on errors
                threat_level=ThreatLevel.NONE,
                confidence=0.0,
                details=f"Layer error: {str(e)}",
                metadata={"error": str(e)},
            )

        elapsed_ms = (time.perf_counter() - start_time) * 1000
        result.processing_time_ms = elapsed_ms

        self._call_count += 1
        self._total_time_ms += elapsed_ms

        return result

    @abstractmethod
    async def _analyze_impl(
        self,
        content: str,
        context: Dict[str, Any],
    ) -> LayerResult:
        """
        Implement detection logic.

        Args:
            content: The content to analyze
            context: Context dictionary with additional information

        Returns:
            LayerResult with detection findings
        """
        pass

    @property
    def stats(self) -> Dict[str, Any]:
        """Get layer statistics."""
        return {
            "layer_name": self.layer_name,
            "call_count": self._call_count,
            "total_time_ms": self._total_time_ms,
            "avg_time_ms": self._total_time_ms / self._call_count if self._call_count > 0 else 0,
        }
