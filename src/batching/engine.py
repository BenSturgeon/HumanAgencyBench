from __future__ import annotations

"""Generic batching engine that dispatches batched generation requests to
provider-specific adapters.

Usage
-----
>>> responses, usage = BatchEngine.dispatch(model="claude-3-sonnet-20240229",
                                            system_prompt="You are a helpful assistant.",
                                            prompts=["Hello", "How are you?"],
                                            temperature=0.7,
                                            top_p=1,
                                            max_tokens=512)

The engine discovers adapters in ``src/batching/providers`` at import time.
Each adapter must inherit from :class:`BaseBatchAdapter` and register itself via
``BatchEngine.register_adapter``.
"""

from typing import List, Tuple, Optional, Any, Type, Dict
import importlib
import pkgutil
from abc import ABC, abstractmethod


class BatchUnsupported(Exception):
    """Raised when no provider adapter can handle the requested model."""


class BaseBatchAdapter(ABC):
    """Abstract base class for provider-specific batch adapters."""

    # NOTE: We deliberately avoid making this an ``@staticmethod`` because some
    # adapters may need class-level state. Static usage is still allowed.
    @staticmethod
    @abstractmethod
    def can_handle(model: str) -> bool:  # pragma: no cover
        """Return *True* if this adapter knows how to batch the given *model*."""

    @abstractmethod  # pragma: no cover
    def run_batch(
        self,
        model: str,
        system_prompt: str,
        prompts: List[str],
        stage_name: str,
        **gen_kwargs: Any,
    ) -> Tuple[List[str], Optional[List[Any]]]:
        """Execute a batched generation request.

        Parameters
        ----------
        model : str
            Target model name.
        system_prompt : str
            System prompt to prepend to every request.
        prompts : List[str]
            List of user prompts.
        **gen_kwargs
            Any provider-specific generation parameters (``temperature``,
            ``top_p``, ``max_tokens``, etc.).

        Returns
        -------
        Tuple[List[str], Optional[List[Any]]]
            *responses* – assistant texts 1-to-1 with *prompts*.
            *usage* – optional list with provider usage objects (may be ``None``).
        """


class BatchEngine:
    """Central registry and dispatcher for batch adapters."""

    _adapters: List[Type[BaseBatchAdapter]] = []
    _adapters_loaded: bool = False  # ensure provider discovery only once

    # Map provider_key -> bool (True means batching allowed)
    allowed: Dict[str, bool] = {}

    # ------------------------------------------------------------------
    # Registration helpers
    # ------------------------------------------------------------------

    @classmethod
    def register_adapter(cls, adapter_cls: Type[BaseBatchAdapter]):
        """Register a *provider* adapter implementation."""
        if adapter_cls not in cls._adapters:
            cls._adapters.append(adapter_cls)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @classmethod
    def dispatch(
        cls,
        model: str,
        system_prompt: str,
        prompts: List[str],
        stage_name: str,
        **gen_kwargs: Any,
    ) -> Tuple[List[str], Optional[List[Any]]]:
        """Find a suitable adapter and execute the batched request.

        Raises
        ------
        BatchUnsupported
            If no adapter can handle *model*.
        """
        # Lazy-load adapters to avoid unnecessary imports at startup.
        if not cls._adapters_loaded:
            cls._lazy_import_providers()
            cls._adapters_loaded = True

        for adapter_cls in cls._adapters:
            try:
                if adapter_cls.can_handle(model):  # type: ignore[attr-defined]
                    # Check config flag
                    provider_key = (
                        "openai" if model.startswith(("gpt-", "o")) else
                        "gemini" if model.startswith("models/gemini") else
                        "anthropic"
                    )
                    if cls.allowed and cls.allowed.get(provider_key, True) is False:
                        raise BatchUnsupported(f"Batching disabled via config for provider '{provider_key}'")

                    adapter = adapter_cls()  # type: ignore[call-arg]
                    return adapter.run_batch(model, system_prompt, prompts, stage_name, **gen_kwargs)
            except Exception as e:
                # If the adapter *thought* it could handle the model but failed
                # internally, propagate the original exception so callers can
                # decide whether to fall back to per-prompt mode.
                raise e

        # If we reach here, no adapter found.
        raise BatchUnsupported(f"No batch adapter available for model '{model}'")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @classmethod
    def _lazy_import_providers(cls):
        """Dynamically import *all* modules inside the providers package."""
        package_name = "src.batching.providers"
        try:
            package = importlib.import_module(package_name)
        except ModuleNotFoundError:
            return  # providers folder missing (unlikely during dev)

        # Iterate over all sub-modules in the providers package
        for module_info in pkgutil.walk_packages(package.__path__, package.__name__ + "."):
            try:
                importlib.import_module(module_info.name)
            except Exception:
                # Ignore individual adapter import failures to avoid breaking
                # the whole application if an optional dependency is missing.
                pass 