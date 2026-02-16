"""Base LLM wrapper -- model-agnostic by design.

Any OpenAI-compatible API can be used by passing a different model name.
Defaults are read from ``settings`` but can be overridden per-instance.
"""

from typing import Any, Dict, List, Optional

from openai import OpenAI

from src.utils.config import settings


class BaseLLM:
    """Thin, model-agnostic wrapper around the OpenAI chat completions API."""

    def __init__(self, model: str | None = None, api_key: str | None = None):
        self.model = model or settings.conversation_model
        self._client = OpenAI(api_key=api_key or settings.openai_api_key)

    def complete(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        **kwargs: Any,
    ) -> Optional[str]:
        """Send *messages* to the model and return the assistant reply text."""
        try:
            resp = self._client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                **kwargs,
            )
            msg = resp.choices[0].message
            return msg.content if msg else None
        except Exception:
            return None
