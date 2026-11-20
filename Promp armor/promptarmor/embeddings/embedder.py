"""Embedding utilities for semantic similarity."""

from typing import List, Optional, Union
import numpy as np
from promptarmor.core.config import EmbeddingProvider


class Embedder:
    """
    Wrapper for embedding models.

    Supports multiple backends:
    - sentence-transformers (local)
    - OpenAI embeddings (API)
    - Vertex AI embeddings (API)
    """

    def __init__(
        self,
        provider: EmbeddingProvider = EmbeddingProvider.SENTENCE_TRANSFORMERS,
        model: str = "all-MiniLM-L6-v2",
    ):
        self.provider = provider
        self.model = model
        self._embedder = None

    async def initialize(self) -> None:
        """Initialize the embedding model."""
        if self.provider == EmbeddingProvider.SENTENCE_TRANSFORMERS:
            try:
                from sentence_transformers import SentenceTransformer
                self._embedder = SentenceTransformer(self.model)
            except ImportError:
                raise ImportError(
                    "sentence-transformers not installed. "
                    "Run: pip install sentence-transformers"
                )

    def encode(
        self,
        texts: Union[str, List[str]],
        normalize: bool = True,
    ) -> np.ndarray:
        """
        Encode texts to embeddings.

        Args:
            texts: Single text or list of texts
            normalize: Whether to L2-normalize embeddings

        Returns:
            Numpy array of embeddings
        """
        if isinstance(texts, str):
            texts = [texts]

        if self._embedder is None:
            raise RuntimeError("Embedder not initialized. Call initialize() first.")

        embeddings = self._embedder.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=normalize,
        )

        return embeddings

    def similarity(
        self,
        text1: str,
        text2: str,
    ) -> float:
        """Calculate cosine similarity between two texts."""
        emb1 = self.encode(text1)
        emb2 = self.encode(text2)
        return float(np.dot(emb1, emb2.T)[0, 0])

    def most_similar(
        self,
        query: str,
        candidates: List[str],
        top_k: int = 5,
    ) -> List[tuple]:
        """
        Find most similar candidates to query.

        Returns:
            List of (candidate, similarity) tuples
        """
        query_emb = self.encode(query)
        candidate_embs = self.encode(candidates)

        similarities = np.dot(query_emb, candidate_embs.T)[0]
        top_indices = np.argsort(similarities)[::-1][:top_k]

        return [
            (candidates[i], float(similarities[i]))
            for i in top_indices
        ]


async def get_embedder(
    provider: EmbeddingProvider = EmbeddingProvider.SENTENCE_TRANSFORMERS,
    model: str = "all-MiniLM-L6-v2",
) -> Embedder:
    """Factory function to create and initialize an embedder."""
    embedder = Embedder(provider=provider, model=model)
    await embedder.initialize()
    return embedder
