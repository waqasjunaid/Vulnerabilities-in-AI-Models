"""Base LLM client interface."""

from abc import ABC, abstractmethod
from typing import Optional, List, Dict, Any


class BaseLLMClient(ABC):
    """Abstract base class for LLM clients."""

    def __init__(self, model: Optional[str] = None, **kwargs):
        self.model = model
        self.config = kwargs

    async def initialize(self) -> None:
        """Initialize the client. Override for async setup."""
        pass

    @abstractmethod
    async def generate(
        self,
        system_prompt: str,
        user_message: str,
        temperature: float = 0.7,
        max_tokens: int = 1024,
        **kwargs,
    ) -> str:
        """
        Generate a response from the LLM.

        Args:
            system_prompt: The system prompt
            user_message: The user's message
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            **kwargs: Additional provider-specific args

        Returns:
            The generated response text
        """
        pass

    @abstractmethod
    async def generate_with_history(
        self,
        system_prompt: str,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 1024,
        **kwargs,
    ) -> str:
        """
        Generate a response with conversation history.

        Args:
            system_prompt: The system prompt
            messages: List of {"role": "user"|"assistant", "content": "..."}
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate

        Returns:
            The generated response text
        """
        pass
