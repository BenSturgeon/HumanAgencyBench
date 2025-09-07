from __future__ import annotations

"""OpenAI Batch API adapter.

Implements :class:`BaseBatchAdapter` for the `/v1/batch` endpoint that went GA
in June-2025.  The adapter uploads all requests in-memory (no temp file
required) by passing the ``requests=[...]`` parameter to
``client.batches.create``.
"""

from typing import List, Tuple, Optional, Any, Dict
import time
import json

from openai import OpenAI

from src.batching.engine import BaseBatchAdapter, BatchEngine, BatchUnsupported


class OpenAIBatchAdapter(BaseBatchAdapter):
    """Adapter for GPT-* / o* models served by OpenAI."""

    @staticmethod
    def can_handle(model: str) -> bool:
        # OpenAI models typically start with "gpt-" or one of the new short "o1",
        # "o3", etc. prefixes.
        return model.startswith("gpt-") or model.startswith("o")

    # ------------------------------------------------------------------
    # Utility helpers
    # ------------------------------------------------------------------

    def _build_requests(
        self,
        model: str,
        system_prompt: str,
        prompts: List[str],
        temperature: Optional[float],
        top_p: Optional[float],
        max_tokens: Optional[int],
    ) -> List[Dict[str, Any]]:
        """Return a list of request dicts suitable for the Batch endpoint."""
        requests: List[Dict[str, Any]] = []
        for i, prompt in enumerate(prompts):
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})

            body: Dict[str, Any] = {
                "model": model,
                "messages": messages,
            }
            # Only include optional params when provided to avoid server errors.
            if temperature is not None:
                body["temperature"] = temperature
            if top_p is not None:
                body["top_p"] = top_p
            if max_tokens is not None:
                body["max_tokens"] = max_tokens

            requests.append(
                {
                    "custom_id": str(i),
                    "method": "POST",
                    "url": "/v1/chat/completions",
                    "body": body,
                }
            )
        return requests

    # ------------------------------------------------------------------
    # Base implementation
    # ------------------------------------------------------------------

    def run_batch(
        self,
        model: str,
        system_prompt: str,
        prompts: List[str],
        stage_name: str,
        **gen_kwargs: Any,
    ) -> Tuple[List[str], Optional[List[Any]]]:
        temperature = gen_kwargs.get("temperature")
        top_p = gen_kwargs.get("top_p")
        max_tokens = gen_kwargs.get("max_tokens")

        # 1. Build request list ------------------------------------------------
        requests = self._build_requests(
            model,
            system_prompt,
            prompts,
            temperature,
            top_p,
            max_tokens,
        )

        # 2. Create batch ------------------------------------------------------
        client = OpenAI()
        try:
            batch = client.batches.create(
                requests=requests,
                completion_window="24h",  # longest allowed; negligible cost diff.
            )
        except TypeError as e:
            # Older openai-python versions (<1.25) don’t support the 'requests' arg
            if "unexpected keyword argument 'requests'" not in str(e):
                raise BatchUnsupported(f"Failed to create OpenAI batch: {e}") from e

            # --- Fallback path: write JSONL file then upload ---------
            import tempfile, os

            jsonl_lines = [json.dumps(r) for r in requests]
            with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".jsonl") as fh:
                fh.write("\n".join(jsonl_lines))
                tmp_path = fh.name

            try:
                input_file = client.files.create(file=open(tmp_path, "rb"), purpose="batch")
                batch = client.batches.create(
                    input_file_id=input_file.id,
                    endpoint="/v1/chat/completions",
                    completion_window="24h",
                )
            except Exception as e2:
                raise BatchUnsupported(f"OpenAI batch fallback via file failed: {e2}") from e2
            finally:
                try:
                    os.unlink(tmp_path)
                except Exception:
                    pass
        except Exception as e:
            raise BatchUnsupported(f"Failed to create OpenAI batch: {e}") from e

        # 3. Poll until finished ----------------------------------------------
        batch_id = batch.id
        print(f"[DEBUG] Submitted OpenAI batch {batch_id} with {len(prompts)} items for stage: {stage_name}")
        start_time = time.time()
        last_status_log = 0.0
        while batch.status not in {"completed", "failed", "expired"}:
            time.sleep(5)
            batch = client.batches.retrieve(batch_id)

            now = time.time()
            if now - last_status_log > 30:
                processed = batch.request_counts.completed if hasattr(batch, "request_counts") and hasattr(batch.request_counts, "completed") else "?"
                total = batch.request_counts.total if hasattr(batch, "request_counts") and hasattr(batch.request_counts, "total") else "?"
                print(f"[INFO] OpenAI batch ({stage_name}) {batch_id} status='{batch.status}' completed={processed}/{total}")
                last_status_log = now
        elapsed = time.time() - start_time
        print(f"[DEBUG] OpenAI batch {batch_id} finished in {elapsed:.1f}s with status='{batch.status}'")

        if batch.status != "completed":
            raise RuntimeError(f"Batch {batch_id} ended with status '{batch.status}'")

        # 4. Download output file --------------------------------------------
        try:
            output_bytes = client.files.content(batch.output_file_id).content
        except Exception as e:
            raise RuntimeError(f"Failed to download batch output: {e}") from e

        lines = output_bytes.decode("utf-8").splitlines()
        results_dicts = [json.loads(l) for l in lines if l.strip()]

        # 5. Map custom_id back to original index -----------------------------
        responses_map: Dict[int, str] = {}
        usage_map: Dict[int, Any] = {}

        for res in results_dicts:
            cid_str = res.get("custom_id")
            if cid_str is None:
                continue
            try:
                cid = int(cid_str)
            except ValueError:
                continue

            # Extract assistant textual content
            try:
                content = (
                    res["response"]["body"]["choices"][0]["message"]["content"].strip()
                )
            except Exception:
                content = "[ERROR PARSING OPENAI BATCH RESULT]"
            responses_map[cid] = content

            # Extract usage object if present
            usage = None
            try:
                usage = res["response"]["body"].get("usage")
            except Exception:
                usage = None
            usage_map[cid] = usage

        # 6. Build ordered lists ---------------------------------------------
        ordered_responses: List[str] = [
            responses_map.get(i, "[ERROR DURING BATCH LLM CHAT]") for i in range(len(prompts))
        ]
        ordered_usage: List[Any] = [usage_map.get(i) for i in range(len(prompts))]

        return ordered_responses, ordered_usage


# Register the adapter
BatchEngine.register_adapter(OpenAIBatchAdapter) 