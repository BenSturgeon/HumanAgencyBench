from __future__ import annotations

"""Utilities for batched LLM requests with transparent hash-cache lookup and
cost-tracking support.

The public function exported here is ``batch_model_response`` which replicates
all behaviours of the existing ``model_response`` helper but sends every cache
MISS to the provider in **one** batched request (currently only Anthropic
Claude models are supported).  Successful responses are written back to the
same ``hash_cache`` directory so subsequent runs can hit the cache.
"""

from typing import List, Tuple, Dict, Optional, Any
import os
import pickle
import hashlib
import time

from src.llms import LLM
from src.utils import hash_cache
from src.cost_tracker import MODEL_COSTS

# ---------------------------------------------------------------------------
# Internal helpers -----------------------------------------------------------
# ---------------------------------------------------------------------------

_CACHE_DIR = "hash_cache"


def _make_cache_filename(
    prompt: str,
    system_prompt: str,
    model: str,
    temperature: float,
    top_p: float,
    max_tokens: int,
    cache_nonce: Any = None,
) -> str:
    """Re-implement the hashing logic of @hash_cache for *model_response*."""

    # The original decorator builds the key as:  [func_name, args, kwargs, nonce]
    args_tuple = (
        prompt,
        system_prompt,
        model,
        temperature,
        top_p,
        max_tokens,
    )
    key_parts = [
        "model_response",  # func.__name__ in evaluate_model.py
        args_tuple,
        {},  # kwargs after stripping extras
        cache_nonce,
    ]

    md5 = hashlib.md5(pickle.dumps(key_parts)).hexdigest()
    return os.path.join(_CACHE_DIR, f"{md5}.pkl")


def _read_cached(prompt, system_prompt, model, temperature, top_p, max_tokens):
    path = _make_cache_filename(prompt, system_prompt, model, temperature, top_p, max_tokens)
    if os.path.exists(path):
        try:
            with open(path, "rb") as fh:
                resp, _sys_prompt = pickle.load(fh)
                return resp
        except Exception:
            # Corrupt cache file – ignore
            return None
    return None


def _write_cached(prompt, system_prompt, model, temperature, top_p, max_tokens, response):
    path = _make_cache_filename(prompt, system_prompt, model, temperature, top_p, max_tokens)
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as fh:
            pickle.dump((response, system_prompt), fh)
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Public API ----------------------------------------------------------------
# ---------------------------------------------------------------------------

def batch_model_response(
    prompts: List[str],
    system_prompt: str,
    model: str,
    temperature: float,
    top_p: float,
    max_tokens: int,
    stage_name: str = "Unknown Stage",
) -> List[str]:
    """Return assistant responses for *prompts* with caching and batch API.

    It first loads any cached responses; remaining prompts are sent in a single
    provider batch (currently only Claude models).  Fresh results are written
    back to the cache *and* logged via CostTracker when available.
    """

    assert len(prompts) > 0, "prompts list is empty"

    # 1. Check cache --------------------------------------------------------
    responses: List[Optional[str]] = [None] * len(prompts)
    uncached_indices: List[int] = []

    for i, p in enumerate(prompts):
        cached = _read_cached(p, system_prompt, model, temperature, top_p, max_tokens)
        if cached is not None:
            responses[i] = cached
        else:
            uncached_indices.append(i)

    if not uncached_indices:
        return [r for r in responses]  # All hits

    # 2. Send batch for misses ---------------------------------------------
    # Try generic BatchEngine first; gracefully fall back to sequential calls
    from src.batching.engine import BatchEngine, BatchUnsupported

    batch_prompts = [prompts[i] for i in uncached_indices]

    try:
        batch_responses, batch_usage = BatchEngine.dispatch(
            model=model,
            system_prompt=system_prompt,
            prompts=batch_prompts,
            stage_name=stage_name,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
        )
        print(f"[DEBUG] Dispatched batch request for model '{model}' ({len(batch_prompts)} prompts).")
        # Ensure we have a usage list aligned with batch_responses length
        if batch_usage is None or len(batch_usage) != len(batch_responses):
            batch_usage = [None] * len(batch_responses)
    except BatchUnsupported as e:
        print(f"[INFO] Falling back to threaded per-prompt calls for model '{model}' – batch unsupported ({e}). Progress bar will appear.")
        batch_responses, batch_usage = [], []

        from concurrent.futures import ThreadPoolExecutor, as_completed
        from tqdm import tqdm as _tqdm

        def _single_call(prompt_str: str):
            single_llm = LLM(model, system_prompt)
            resp = single_llm.chat(prompt_str, temperature=temperature, top_p=top_p, max_tokens=max_tokens)
            if isinstance(resp, dict):
                return resp.get("content", ""), resp.get("usage")
            return resp, None

        max_workers = min(16, len(batch_prompts))
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_idx = {executor.submit(_single_call, p): idx for idx, p in enumerate(batch_prompts)}
            for future in _tqdm(as_completed(future_to_idx), total=len(future_to_idx), desc=f"Threaded fallback ({model})"):
                idx = future_to_idx[future]
                content, usage_obj = future.result()
                batch_responses.append((idx, content))
                batch_usage.append((idx, usage_obj))

        # restore original order
        batch_responses_sorted = [None]*len(batch_prompts)
        batch_usage_sorted = [None]*len(batch_prompts)
        for idx, txt in batch_responses:
            batch_responses_sorted[idx]=txt
        for idx, u in batch_usage:
            batch_usage_sorted[idx]=u
        batch_responses, batch_usage = batch_responses_sorted, batch_usage_sorted

    except Exception as e:
        print(f"[WARN] Batch engine error for model '{model}': {e}. Falling back to sequential calls.")
        llm = LLM(model, system_prompt)
        batch_responses = []
        batch_usage = []
        for p in batch_prompts:
            r = llm.chat(p, temperature=temperature, top_p=top_p, max_tokens=max_tokens)
            if isinstance(r, dict):
                batch_responses.append(r.get("content", ""))
                batch_usage.append(r.get("usage"))
            else:
                batch_responses.append(r)
                batch_usage.append(None)

    # Sanity check
    assert len(batch_responses) == len(uncached_indices), "Mismatch in batch response length"

    # 3. Fill responses list & write cache ---------------------------------
    for local_idx, global_idx in enumerate(uncached_indices):
        resp_text = batch_responses[local_idx]
        usage_obj = batch_usage[local_idx] if local_idx < len(batch_usage) else None
        responses[global_idx] = resp_text
        _write_cached(
            prompts[global_idx],
            system_prompt,
            model,
            temperature,
            top_p,
            max_tokens,
            resp_text,
        )

        # 4. Cost tracking per prompt (optional) ---------------------------
        # Cost tracking – fetch tracker at runtime to ensure we get the
        # instance created by wrap_llms_for_cost_tracking() *after* imports.
        try:
            import importlib
            cti = importlib.import_module("src.cost_tracking_integration")
            tracker = getattr(cti, "cost_tracker_instance", None)
            if tracker is not None:
                messages_for_tracking = [{"role": "user", "content": prompts[global_idx]}]
                tracker.track_chat_completion(
                    model=model,
                    messages=messages_for_tracking,
                    response=resp_text,
                    metadata={"batched": True},
                    cached_input=False,
                    usage=usage_obj,
                )
        except Exception:
            pass

    return [r for r in responses] 