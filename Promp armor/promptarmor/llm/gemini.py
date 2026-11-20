"""Google Gemini client implementation."""

import os
from typing import Optional, List, Dict, Any

from promptarmor.llm.base import BaseLLMClient


class GeminiClient(BaseLLMClient):
    """Client for Google's Gemini models."""

    DEFAULT_MODEL = "gemini-1.5-flash"

    def __init__(self, model: Optional[str] = None, **kwargs):
        super().__init__(model or self.DEFAULT_MODEL, **kwargs)
        self.client = None

    async def initialize(self) -> None:
        """Initialize the Gemini client."""
        try:
            import google.generativeai as genai
            api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
            if not api_key:
                raise ValueError("GOOGLE_API_KEY or GEMINI_API_KEY environment variable not set")
            genai.configure(api_key=api_key)
            self.client = genai.GenerativeModel(self.model)
        except ImportError:
            raise ImportError("google-generativeai package not installed. Run: pip install google-generativeai")

    async def generate(
        self,
        system_prompt: str,
        user_message: str,
        temperature: float = 0.7,
        max_tokens: int = 1024,
        **kwargs,
    ) -> str:
        """Generate a response using Gemini."""
        if self.client is None:
            await self.initialize()

        # Gemini combines system + user in the prompt
        full_prompt = f"{system_prompt}\n\nUser: {user_message}"

        response = await self.client.generate_content_async(
            full_prompt,
            generation_config={
                "temperature": temperature,
                "max_output_tokens": max_tokens,
            },
        )

        return response.text

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

        # Build conversation for Gemini
        parts = [system_prompt, ""]
        for msg in messages:
            role = "User" if msg["role"] == "user" else "Assistant"
            parts.append(f"{role}: {msg['content']}")

        full_prompt = "\n\n".join(parts)

        response = await self.client.generate_content_async(
            full_prompt,
            generation_config={
                "temperature": temperature,
                "max_output_tokens": max_tokens,
            },
        )

        return response.text
