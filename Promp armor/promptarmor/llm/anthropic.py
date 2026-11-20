"""Anthropic Claude client implementation."""

import os
from typing import Optional, List, Dict, Any

from promptarmor.llm.base import BaseLLMClient


class AnthropicClient(BaseLLMClient):
    """Client for Anthropic's Claude models."""

    DEFAULT_MODEL = "claude-3-sonnet-20240229"

    def __init__(self, model: Optional[str] = None, **kwargs):
        super().__init__(model or self.DEFAULT_MODEL, **kwargs)
        self.client = None

    async def initialize(self) -> None:
        """Initialize the Anthropic client."""
        try:
            import anthropic
            api_key = os.getenv("ANTHROPIC_API_KEY")
            if not api_key:
                raise ValueError("ANTHROPIC_API_KEY environment variable not set")
            self.client = anthropic.AsyncAnthropic(api_key=api_key)
        except ImportError:
            raise ImportError("anthropic package not installed. Run: pip install anthropic")

    async def generate(
        self,
        system_prompt: str,
        user_message: str,
        temperature: float = 0.7,
        max_tokens: int = 1024,
        **kwargs,
    ) -> str:
        """Generate a response using Claude."""
        if self.client is None:
            await self.initialize()

        response = await self.client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            temperature=temperature,
            system=system_prompt,
            messages=[
                {"role": "user", "content": user_message}
            ],
            **kwargs,
        )

        return response.content[0].text

    async def generate_with_history(
        self,
        system_prompt: str,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 1024,
        **kwargs,
    ) -> str:
        """Generate a response with conversation history."""
        if self.client is None:
            await self.initialize()

        response = await self.client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            temperature=temperature,
            system=system_prompt,
            messages=messages,
            **kwargs,
        )

        return response.content[0].text
