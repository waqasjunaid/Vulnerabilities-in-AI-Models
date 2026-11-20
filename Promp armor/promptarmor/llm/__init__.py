"""LLM client implementations for PromptArmor."""

from typing import Optional
from promptarmor.core.config import LLMProvider
from promptarmor.llm.base import BaseLLMClient
from promptarmor.llm.anthropic import AnthropicClient
from promptarmor.llm.openai import OpenAIClient
from promptarmor.llm.gemini import GeminiClient


async def get_llm_client(
    provider: LLMProvider,
    model: Optional[str] = None,
    **kwargs,
) -> BaseLLMClient:
    """
    Factory function to get an LLM client.

    Args:
        provider: The LLM provider to use
        model: Optional model override
        **kwargs: Additional client configuration

    Returns:
        Initialized LLM client
    """
    clients = {
        LLMProvider.ANTHROPIC: AnthropicClient,
        LLMProvider.OPENAI: OpenAIClient,
        LLMProvider.GEMINI: GeminiClient,
    }

    client_class = clients.get(provider)
    if not client_class:
        raise ValueError(f"Unknown provider: {provider}")

    client = client_class(model=model, **kwargs)
    await client.initialize()
    return client


__all__ = [
    "get_llm_client",
    "BaseLLMClient",
    "AnthropicClient",
    "OpenAIClient",
    "GeminiClient",
]
