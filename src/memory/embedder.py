"""Embedding generation via OpenAI (synchronous for PoC simplicity)."""

from typing import List

from openai import OpenAI

from src.utils.config import settings
from src.utils.logger import get_logger

log = get_logger(__name__)

EMBEDDING_DIM = 1536  # text-embedding-3-small


class Embedder:
    """Generate vector embeddings using the OpenAI embeddings API."""

    def __init__(self, model: str | None = None, api_key: str | None = None):
        self.model = model or settings.embedding_model
        self._client = OpenAI(api_key=api_key or settings.openai_api_key)

    @property
    def dimension(self) -> int:
        return EMBEDDING_DIM

    def embed_text(self, text: str) -> List[float]:
        """Embed a single text string and return the vector."""
        resp = self._client.embeddings.create(model=self.model, input=text)
        return resp.data[0].embedding

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Embed a batch of texts, preserving input order."""
        if not texts:
            return []
        resp = self._client.embeddings.create(model=self.model, input=texts)
        by_index = {d.index: d.embedding for d in resp.data}
        return [by_index[i] for i in range(len(texts))]
