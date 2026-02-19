"""Conversation LLM -- generates user-facing responses from context.

Default model: gpt-5.2 (configurable via OPENAI_CONVERSATION_MODEL).
"""

from typing import Dict, List, Optional

from src.llm.base import BaseLLM
from src.utils.config import settings
from src.utils.logger import get_logger

log = get_logger(__name__)

RESPONSE_PROMPT = """You are a helpful AI assistant. Answer the user's question based on the provided context.

User Question: {query}

Context:
{context}

Sources:
{sources}

Instructions:
- Provide a clear, accurate answer
- Cite sources using [1], [2] format
- If context is insufficient, say so honestly
- Keep responses concise but complete

Answer:"""


class ConversationLLM(BaseLLM):
    """Generates final responses shown to the user."""

    def __init__(self, model: str | None = None, **kwargs):
        super().__init__(model=model or settings.conversation_model, **kwargs)

    def generate_response(
        self,
        query: str,
        context: List[str],
        metadata: List[Dict],
    ) -> Optional[str]:
        """Build the prompt, call the model, and return the answer string."""
        numbered_ctx = "\n".join(
            f"[{i+1}] {chunk}" for i, chunk in enumerate(context)
        )
        sources_block = "\n".join(
            f"[{i+1}] {m.get('url', m.get('source_url', 'unknown'))}"
            for i, m in enumerate(metadata)
        )
        user_content = RESPONSE_PROMPT.format(
            query=query,
            context=numbered_ctx,
            sources=sources_block,
        )
        messages = [{"role": "user", "content": user_content}]
        return self.complete(messages, temperature=0.4)
