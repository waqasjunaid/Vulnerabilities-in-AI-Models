"""OpenAI client implementation."""

import os
from typing import Optional, List, Dict, Any

from promptarmor.llm.base import BaseLLMClient


class OpenAIClient(BaseLLMClient):
    """Client for OpenAI's GPT models."""

    DEFAULT_MODEL = "gpt-4-turbo-preview"

    def __init__(self, model: Optional[str] = None, **kwargs):
        super().__init__(model or self.DEFAULT_MODEL, **kwargs)
        self.client = None

    async def initialize(self) -> None:
        """Initialize the OpenAI client."""
        try:
            from openai import AsyncOpenAI
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OPENAI_API_KEY environment variable not set")
            self.client = AsyncOpenAI(api_key=api_key)
        except ImportError:
            raise ImportError("openai package not installed. Run: pip install openai")

    async def generate(
        self,
        system_prompt: str,
        user_message: str,
        temperature: float = 0.7,
        max_tokens: int = 1024,
        **kwargs,
    ) -> str:
        """Generate a response using GPT."""
        if self.client is None:
            await self.initialize()

        response = await self.client.chat.completions.create(
            model=self.model,
            max_tokens=max_tokens,
            temperature=temperature,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            **kwargs,
        )

        return response.choices[0].message.content

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

        full_messages = [{"role": "system", "content": system_prompt}] + messages

        response = await self.client.chat.completions.create(
            model=self.model,
            max_tokens=max_tokens,
            temperature=temperature,
            messages=full_messages,
            **kwargs,
        )

        return response.choices[0].message.content
