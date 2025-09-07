from __future__ import annotations

"""Anthropic Claude batch adapter using the SDK *Message Batches* API.

This delegates to the existing ``AnthropicLLM.chat_batch`` helper implemented
in :pymod:`src.llms`.  It therefore inherits all rate-limiting and error-
handling logic already present in that function.
"""

from typing import List, Tuple, Optional, Any

from src.batching.engine import BaseBatchAdapter, BatchEngine, BatchUnsupported
from src.llms import LLM, AnthropicLLM


class AnthropicBatchAdapter(BaseBatchAdapter):
    """Adapter for Anthropic Claude models ("claude-*")."""

    @staticmethod
    def can_handle(model: str) -> bool:
        # All Claude models start with "claude" per internal conventions.
        return model.startswith("claude")

    def run_batch(
        self,
        model: str,
        system_prompt: str,
        prompts: List[str],
        stage_name: str,
        **gen_kwargs: Any,
    ) -> Tuple[List[str], Optional[List[Any]]]:
        llm = LLM(model, system_prompt)

        # Defensive check – the LLM wrapper *should* pick AnthropicLLM here.
        if not isinstance(llm.llm, AnthropicLLM):
            raise BatchUnsupported(
                f"Model '{model}' is not handled by AnthropicBatchAdapter."
            )

        # ``AnthropicLLM.chat_batch`` supports temperature, top_p, max_tokens.
        responses = llm.llm.chat_batch(
            prompts,
            stage_name=stage_name,
            temperature=gen_kwargs.get("temperature"),
            max_tokens=gen_kwargs.get("max_tokens"),
            top_p=gen_kwargs.get("top_p"),
        )

        # The helper currently doesn't expose usage data. Return ``None``.
        return responses, None


# Register this adapter with the global BatchEngine registry.
BatchEngine.register_adapter(AnthropicBatchAdapter) 