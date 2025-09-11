from abc import ABC, abstractmethod
import time
from threading import Lock
from typing import Any, Dict, Optional, Callable, List
import os

from openai import OpenAI
import anthropic
from groq import Groq
import replicate
import google.generativeai as genai


class ABSTRACT_LLM(ABC):
    def __init__(self, system_prompt: str):
        self.system_prompt = system_prompt
        self.messages = []
        if system_prompt:
            self.messages.append({"role": "system", "content": system_prompt})

    @abstractmethod
    def chat(self, text: str) -> str:
        pass


class AvailableModelsCache:

    """A mixin for caching available models at the class level to avoid repeated API calls."""

    _model_cache = {}
    _model_cache_locks = {}

    @classmethod
    def model_cache_get(cls, tag: str, fetch_fn: Callable[[], List[str]]) -> List[str]:
        if tag not in cls._model_cache_locks:
            cls._model_cache_locks[tag] = Lock()

        with cls._model_cache_locks[tag]:
            if tag in cls._model_cache:
                return cls._model_cache[tag]
            models = fetch_fn()
            cls._model_cache[tag] = models
            return models


class RateLimitedLLM:

    """A mixin for rate limiting LLM requests."""

    _rate_limiting = {}

    def rate_limit_wait(self, tag, req_per_min: float):

        interval = 150 / req_per_min

        if tag not in self._rate_limiting:
            self._rate_limiting[tag] = {"last_request": 0, "lock": Lock()}

        while True:
            with self._rate_limiting[tag]["lock"]:
                sleep = time.monotonic() - self._rate_limiting[tag]["last_request"] < interval

            if sleep:
                time.sleep(1)
            else:
                with self._rate_limiting[tag]["lock"]:
                    self._rate_limiting[tag]["last_request"] = time.monotonic()

                break


class OpenAILLM(ABSTRACT_LLM, AvailableModelsCache):
    def __init__(self, model, system_prompt, base_url=None, api_key=None) -> None:
        super().__init__(system_prompt)
        self.base_url = base_url
        self.api_key = api_key
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
        )
        self.model = model
        # Check if model is from the OpenAI "o" series (o1, o3, o4, etc.)
        self.is_o1_model = model and (
            model.startswith("o1") or 
            model.startswith("o3") or 
            model.startswith("o4") or
            "o1-" in model or
            "o3-" in model or
            "o4-" in model
        )

    def chat(
        self,
        prompt,
        temperature=None,
        max_tokens=None,
        top_p=None,
        return_json=False,
        return_logprobs=False,
    ):
        if self.is_o1_model:
            messages = [msg for msg in self.messages if msg["role"] != "system"]

            if self.system_prompt:
                prompt = f"{self.system_prompt}\n{prompt}"

            messages.append({"role": "user", "content": prompt})

            params = {
                "model": self.model,
                "messages": messages,
            }
            
            # Add optional parameters only if they are not None
            
            params["temperature"] = 1

            response = self.client.chat.completions.create(**params)
            response_text = response.choices[0].message.content.strip()

            self.messages.append({"role": "user", "content": prompt})
            self.messages.append({"role": "assistant", "content": response_text})

            # Return both text and usage so downstream cost trackers can see full token counts
            return {
                "content": response_text,
                "usage": response.usage,
            }
        else:
            self.messages.append({"role": "user", "content": prompt})
            params = {
                "messages": self.messages,
                "model": self.model,
            }
            
            # Add optional parameters only if they are not None
            if temperature is not None:
                params["temperature"] = temperature
            if max_tokens is not None:
                params["max_tokens"] = max_tokens
            if top_p is not None:
                params["top_p"] = top_p
            if return_json:
                params["response_format"] = {"type": "json_object"}
            
            if return_logprobs:
                params.update({"logprobs": True, "top_logprobs": 5})

            response = self.client.chat.completions.create(**params)
            response_text = response.choices[0].message.content.strip()
            self.messages.append({"role": "assistant", "content": response_text})

            if return_logprobs:
                return {
                    "content": response.choices[0].message.content,
                    "logprobs": response.choices[0].logprobs,
                    "usage": response.usage,
                }
            else:
                return {
                    "content": response_text,
                    "usage": response.usage,
                }

    @classmethod
    def get_available_models(cls):
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            return []
        
        try:
            return cls.model_cache_get(
                "openai_models",
                lambda: [model.id for model in OpenAI().models.list().data]
            )
        except Exception:
            return []


class GrokLLM(OpenAILLM):
    def __init__(self, model: str, system_prompt: str) -> None:
        super().__init__(model, system_prompt, base_url="https://api.x.ai/v1")
        self.client = OpenAI(
            api_key=os.getenv("GROK_API_KEY"),
            base_url="https://api.x.ai/v1"
        )
        self.model = model
        self.is_o1_model = False

    @classmethod
    def get_available_models(cls) -> List[str]:
        api_key = os.getenv("GROK_API_KEY")
        if not api_key:
            return []
        
        client = OpenAI(
            api_key=api_key,
            base_url="https://api.x.ai/v1"
        )
    
        try:
            return cls.model_cache_get(
                "grok_models",
                lambda: [model.id for model in client.models.list().data]
            )
        except Exception:
            return []


class DeepSeekLLM(OpenAILLM):
    def __init__(self, model: str, system_prompt: str) -> None:
        super().__init__(model, system_prompt, base_url="https://api.deepseek.ai/v1")
        self.client = OpenAI(
            api_key=os.getenv("DEEPSEEK_API_KEY"),
            base_url="https://api.deepseek.com/v1"
        )
        self.model = model
        self.is_o1_model = False

    @staticmethod
    def get_available_models() -> List[str]:
        return [
            "deepseek-chat",
            "deepseek-reasoner"
        ]


class AnthropicLLM(ABSTRACT_LLM, RateLimitedLLM):
    def __init__(self, model, system_prompt) -> None:
        super().__init__(system_prompt)
        self.messages = []
        self.system_prompt = system_prompt
        self.client = anthropic.Anthropic()
        self.model = model

    def chat(
        self, prompt, temperature, max_tokens, top_p, return_json, return_logprobs=False
    ):
        

        if return_logprobs:
            raise NotImplementedError(
                "Anthropic LLM not implemented for return_logprobs"
            )

        self.messages.append(
            {"role": "user", "content": [{"type": "text", "text": prompt}]}
        )
        response = self.client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            system=self.system_prompt,
            messages=self.messages,
        )
        response_text = response.content[0].text
        self.messages.append(
            {"role": "assistant", "content": [{"type": "text", "text": response_text}]}
        )

        return response_text

    def chat_batch(
        self,
        prompts: List[str],
        stage_name: str = "Unknown Stage",
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        top_p: Optional[float] = None,
        **kwargs,
    ) -> List[str]:
        """Send a batch of prompts using Anthropic's Message Batches API.

        Parameters
        ----------
        prompts : List[str]
            A list of user prompts to send.
        temperature, max_tokens, top_p : optional
            Usual generation parameters.

        Returns
        -------
        List[str]
            A list of responses corresponding 1-to-1 with `prompts`.
        """

        # Guard: empty prompts
        if not prompts:
            return []

        # Respect existing rate limit logic (up to 30 req/min per model)
        # We treat the batch as one request.
        self.rate_limit_wait(self.model, 30)

        # Lazy import of types – older SDK versions may not ship them, so we
        # fall back to per-prompt chat() calls if batch types are missing.
        try:
            from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
            from anthropic.types.messages.batch_create_params import Request as AnthropicRequest
        except ImportError:
            # Silently degrade to per-prompt chat() calls if batch types are missing
            return [
                self.chat(p, temperature=temperature, max_tokens=max_tokens, top_p=top_p)
                for p in prompts
            ]

        # Build request objects
        req_temperature = temperature if temperature is not None else 1
        req_top_p = top_p if top_p is not None else 1
        req_max_tokens = max_tokens if max_tokens is not None else 1024

        anthropic_reqs = []
        for idx, prompt in enumerate(prompts):
            # Compose messages: optional system prompt + user prompt
            messages_payload = [
                {
                    "role": "user",
                    "content": prompt,
                }
            ]

            params = MessageCreateParamsNonStreaming(
                model=self.model,
                messages=messages_payload,
                max_tokens=req_max_tokens,
                temperature=req_temperature,
                top_p=req_top_p,
                system=self.system_prompt,
            )

            anthropic_reqs.append(
                AnthropicRequest(
                    custom_id=str(idx),
                    params=params,
                )
            )

        # Fire off the batch
        print(f"[DEBUG] Submitting Anthropic batch with {len(prompts)} prompts to model '{self.model}'")
        batch = self.client.messages.batches.create(requests=anthropic_reqs)

        # Poll until finished (Anthropic docs: may take up to 24h, but we'll
        # poll every 2s; caller can add timeout externally).
        batch_id = batch.id
        start_time = time.time()
        last_status_log = 0.0
        while batch.processing_status == "in_progress":
            time.sleep(2)
            batch = self.client.messages.batches.retrieve(batch_id)

            # Emit a heartbeat every ~30 s so users see progress
            now = time.time()
            if now - last_status_log > 30:
                completed = getattr(batch, 'completed_requests', 0)
                total = getattr(batch, 'total_requests', '??')
                print(f"[INFO] Anthropic batch ({stage_name}) {batch_id} status='{batch.processing_status}' completed={completed}/{total}")
                last_status_log = now

        print(
            f"[DEBUG] Batch ({stage_name}) {batch_id} finished with status '{batch.processing_status}' in {time.time() - start_time:.1f}s"
        )

        # At this point processing ended (succeeded/errored etc.)
        # Fetch results – use SDK helper if present, else stream JSONL.
        results_objs = []
        try:
            # Newer SDKs expose `.results()` which yields MessageBatchIndividualResponse objects
            results_iter = self.client.messages.batches.results(batch_id)
            results_objs = list(results_iter)
        except Exception:
            # Fall back to downloading JSONL from results_url (dicts)
            import json, requests as _requests

            if not getattr(batch, "results_url", None):
                raise RuntimeError("Batch processing ended but no results_url present.")

            resp = _requests.get(batch.results_url)
            resp.raise_for_status()
            results_objs = [json.loads(l) for l in resp.text.splitlines() if l.strip()]

        # Normalize each result into a dict
        results_lines = []
        for obj in results_objs:
            if isinstance(obj, dict):
                results_lines.append(obj)
            else:
                # Pydantic model – use model_dump if available, else vars()
                if hasattr(obj, "model_dump"):
                    results_lines.append(obj.model_dump(exclude_none=True))
                else:
                    results_lines.append(vars(obj))
 
        for item in results_lines:
            res = item.get("result")
            if res is not None and not isinstance(res, dict) and hasattr(res, "model_dump"):
                item["result"] = res.model_dump(exclude_none=True)

        # Helper to robustly extract assistant text from varying result schemas
        def _extract_text(obj: dict):
            # Case 1: modern schema => obj['message']['content'][0]['text']
            if isinstance(obj.get("message"), dict):
                content = obj["message"].get("content")
                if isinstance(content, list) and content and "text" in content[0]:
                    return content[0]["text"]

            # Case 2: some SDKs nest under 'response'
            if isinstance(obj.get("response"), dict):
                content = obj["response"].get("content")
                if isinstance(content, list) and content and "text" in content[0]:
                    return content[0]["text"]

            # Case 2b: new batch schema => obj['result']['message']['content'][0]['text']
            if isinstance(obj.get("result"), dict):
                res = obj["result"]
                if res.get("type") == "succeeded" and isinstance(res.get("message"), dict):
                    content = res["message"].get("content")
                    if isinstance(content, list) and content and "text" in content[0]:
                        return content[0]["text"]

            # Case 3: flat 'content' list
            if isinstance(obj.get("content"), list):
                content = obj["content"]
                if content and isinstance(content[0], dict) and "text" in content[0]:
                    return content[0]["text"]

            # Case 4: flat 'content' string (unlikely for chat)
            if isinstance(obj.get("content"), str):
                return obj["content"]

            return None

        # Map custom_id back to index → response text
        responses_map = {}
        for item in results_lines:
            # The schema uses either `custom_id` or `request.custom_id` depending on SDK.
            cid = item.get("custom_id") or item.get("request", {}).get("custom_id")
            assistant_text = _extract_text(item)

            if assistant_text is None:
                print(f"[WARN] Unable to parse assistant text from batch result (keys={list(item.keys())}).")
                assistant_text = "[ERROR PARSING BATCH RESULT]"

            responses_map[int(cid)] = assistant_text

        ordered_responses = [responses_map.get(i, "[ERROR DURING BATCH LLM CHAT]") for i in range(len(prompts))]

        for p, r in zip(prompts, ordered_responses):
            self.messages.append({"role": "user", "content": p})
            self.messages.append({"role": "assistant", "content": r})

        return ordered_responses

    @staticmethod
    def get_available_models():
        return [ 
            "claude-3-7-sonnet-20250219",
            "claude-3-5-sonnet-20240620",
            "claude-3-opus-20240229",
            "claude-3-sonnet-20240229",
            "claude-3-haiku-20240307",
            "claude-3-5-sonnet-20241022",
            "claude-3-5-haiku-20241022",
            "claude-sonnet-4-20250514",
            "claude-opus-4-20250514"
        ]


class GroqLLM(ABSTRACT_LLM, AvailableModelsCache, RateLimitedLLM):
    def __init__(self, model: str, system_prompt: str) -> None:
        super().__init__(system_prompt)
        self.client = Groq()
        self.model = model

    def chat(
        self,
        prompt: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        top_p: Optional[float] = None,
        return_json: bool = False,
        return_logprobs: bool = False
    ) -> str:

        self.rate_limit_wait(self.model, 15)

        if return_logprobs:
            raise NotImplementedError("Groq LLM not implemented for return_logprobs")
        self.messages.append({"role": "user", "content": prompt})
        response = self.client.chat.completions.create(
            messages=self.messages,
            model=self.model,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            response_format={"type": "json_object"} if return_json else None,
        )
        response_text = response.choices[0].message.content
        self.messages.append({"role": "assistant", "content": response_text})
        return response_text

    @classmethod
    def get_available_models(cls) -> List[str]:
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            return []
        
        try:
            return cls.model_cache_get(
                "groq_models",
                lambda: [model.id for model in Groq().models.list().data]
            )
        except Exception:
            return []


class ReplicateLLM(ABSTRACT_LLM):
    def __init__(self, model: str, system_prompt: str) -> None:
        super().__init__(system_prompt)
        self.client = replicate.Client()
        self.model = model

    def chat(
        self,
        prompt: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        top_p: Optional[float] = None,
        return_json: bool = False,
        return_logprobs: bool = False
    ) -> str:
        if return_logprobs:
            raise NotImplementedError("Replicate LLM not implemented for return_logprobs")

        self.messages.append({"role": "user", "content": prompt})

        input_data: Dict[str, Any] = {
            "prompt": prompt,
            "system_prompt": self.system_prompt,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "top_p": top_p,
        }

        input_data = {k: v for k, v in input_data.items() if v is not None}

        output = self.client.run(
            self.model,
            input=input_data
        )

        response_text = "".join(output)

        self.messages.append({"role": "assistant", "content": response_text})

        return response_text

    @staticmethod
    def get_available_models() -> list:
        return [
            "meta/meta-llama-3.1-405b-instruct",
            "meta/meta-llama-3-70b-instruct",
            "meta/llama-4-scout-instruct",
            "meta/llama-4-maverick-instruct"
        ]


class GeminiLLM(ABSTRACT_LLM, AvailableModelsCache, RateLimitedLLM):

    rate_limiting = {}

    def __init__(self, model: str, system_prompt: str) -> None:
        super().__init__(system_prompt)
        self.model = model
        genai.configure()

        if model not in self.rate_limiting:
            self.rate_limiting[model] = {"last_request_time": 0, "lock": Lock()}

    def chat(
        self,
        prompt: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        top_p: Optional[float] = None,
        return_json: bool = False,
        return_logprobs: bool = False
    ) -> str:

        self.rate_limit_wait(self.model, 150)

        if return_logprobs:
            raise NotImplementedError("Gemini LLM not implemented for return_logprobs")

        self.messages.append({"role": "user", "content": prompt})

        generation_config = {
            "temperature": temperature,
            "top_p": top_p,
            "max_output_tokens": max_tokens,
        }
        generation_config = {k: v for k, v in generation_config.items() if v is not None}

        model = genai.GenerativeModel(model_name=self.model)
        chat = model.start_chat(history=[])

        if self.system_prompt:
            chat.send_message(self.system_prompt) 

        response = chat.send_message(
            prompt,
            generation_config=generation_config
        )

        # Gemini may sometimes return a response whose candidate has no textual `Part`,
        # which causes `response.text` to raise an exception (finish_reason != 0, etc.).
        try:
            response_text = response.text
        except Exception:
            response_text = ""
            try:
                if hasattr(response, "candidates"):
                    for cand in response.candidates:
                        content = getattr(cand, "content", None)
                        if content and hasattr(content, "parts"):
                            for part in content.parts:
                                part_text = getattr(part, "text", None)
                                if part_text:
                                    response_text += part_text
            except Exception as e:
                # If extraction still fails, default to empty string to avoid crashing.
                response_text = ""
                print(f"Error extracting Gemini response text: {e}, response: {response}")

        self.messages.append({"role": "assistant", "content": response_text})

        return response_text

    @classmethod
    def get_available_models(cls) -> List[str]:
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            return []
        
        try:
            genai.configure()
            return cls.model_cache_get(
                "gemini_models",
                lambda: [
                    model.name for model in genai.list_models()
                    if model.name.startswith("models/gemini")
                ]
            )
        except Exception:
            return []


class OpenRouterLLM(ABSTRACT_LLM, AvailableModelsCache, RateLimitedLLM):
    """OpenRouter LLM implementation using OpenAI-compatible API."""
    
    def __init__(self, model: str, system_prompt: str) -> None:
        super().__init__(system_prompt)
        self.model = model
        self.client = OpenAI(
            api_key=os.getenv("OPENROUTER_API_KEY"),
            base_url="https://openrouter.ai/api/v1",
            default_headers={
                "HTTP-Referer": "https://github.com/agency_evaluations",
                "X-Title": "Agency Evaluations"
            }
        )
        
    def chat(
        self,
        prompt: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        top_p: Optional[float] = None,
        return_json: bool = False,
        return_logprobs: bool = False
    ) -> Dict[str, Any]:
        """Send a single chat request to OpenRouter."""
        
        # Rate limiting - adjust based on model
        self.rate_limit_wait(self.model, 60)  # Default 60 req/min
        
        self.messages.append({"role": "user", "content": prompt})
        
        params = {
            "model": self.model,
            "messages": self.messages,
        }
        
        # Add optional parameters
        if temperature is not None:
            params["temperature"] = temperature
        if max_tokens is not None:
            params["max_tokens"] = max_tokens
        if top_p is not None:
            params["top_p"] = top_p
        if return_json:
            params["response_format"] = {"type": "json_object"}
        if return_logprobs:
            params["logprobs"] = True
            params["top_logprobs"] = 5
            
        try:
            response = self.client.chat.completions.create(**params)
            response_text = response.choices[0].message.content.strip()
            self.messages.append({"role": "assistant", "content": response_text})
            
            result = {
                "content": response_text,
                "usage": response.usage,
            }
            
            if return_logprobs and hasattr(response.choices[0], 'logprobs'):
                result["logprobs"] = response.choices[0].logprobs
                
            return result
            
        except Exception as e:
            # Handle errors gracefully
            error_msg = f"[ERROR DURING LLM CHAT]: {str(e)}"
            self.messages.append({"role": "assistant", "content": error_msg})
            return {
                "content": error_msg,
                "usage": None,
            }
    
    @classmethod
    def get_available_models(cls) -> List[str]:
        """Get available models from OpenRouter."""
        def fetch_models():
            import requests
            try:
                response = requests.get("https://openrouter.ai/api/v1/models")
                if response.status_code == 200:
                    models_data = response.json()
                    return [model["id"] for model in models_data.get("data", [])]
                else:
                    return cls._get_default_models()
            except Exception:
                return cls._get_default_models()
                
        return cls.model_cache_get("openrouter_models", fetch_models)
    
    @staticmethod
    def _get_default_models() -> List[str]:
        """Default list of known OpenRouter models with provider prefixes."""
        return [
            # OpenAI models
            "openai/gpt-4-turbo",
            "openai/gpt-4-turbo-preview", 
            "openai/gpt-4",
            "openai/gpt-3.5-turbo",
            "openai/gpt-4o",
            "openai/gpt-4o-mini",
            "openai/o1-preview",
            "openai/o1-mini",
            
            "anthropic/claude-3-opus",
            "anthropic/claude-3-sonnet",
            "anthropic/claude-3-haiku",
            "anthropic/claude-3.5-sonnet",
            "anthropic/claude-2.1",
            "anthropic/claude-2",
            "anthropic/claude-instant-1.2",
            
            # Google models
            "google/gemini-pro",
            "google/gemini-pro-1.5",
            "google/gemini-flash-1.5",
            "google/gemma-2-27b-it",
            "google/gemma-2-9b-it",
            
            # Meta models
            "meta-llama/llama-3-70b-instruct",
            "meta-llama/llama-3-8b-instruct",
            "meta-llama/llama-3.1-405b-instruct",
            "meta-llama/llama-3.1-70b-instruct",
            "meta-llama/llama-3.1-8b-instruct",
            "meta-llama/llama-3.2-90b-vision-instruct",
            "meta-llama/llama-3.2-11b-vision-instruct",
            "meta-llama/llama-3.2-3b-instruct",
            "meta-llama/llama-3.2-1b-instruct",
            
            # Mistral models
            "mistralai/mistral-large",
            "mistralai/mistral-medium",
            "mistralai/mixtral-8x22b-instruct",
            "mistralai/mixtral-8x7b-instruct",
            "mistralai/mistral-7b-instruct",
            "mistralai/mistral-small",
            
            # Other popular models
            "deepseek/deepseek-chat",
            "deepseek/deepseek-coder",
            "nousresearch/hermes-3-llama-3.1-405b",
            "qwen/qwen-2.5-72b-instruct",
            "qwen/qwen-2.5-32b-instruct",
            "cohere/command-r-plus",
            "cohere/command-r",
            "x-ai/grok-beta",
            "perplexity/llama-3.1-sonar-large-128k-chat",
            "perplexity/llama-3.1-sonar-small-128k-chat",
        ]


class LLM(ABSTRACT_LLM):
    def __init__(self, model, system_prompt) -> None:
        self.system_prompt = system_prompt
        self.model = model
        
        if model in GeminiLLM.get_available_models():
            self.llm = GeminiLLM(model, system_prompt)
        elif model in AnthropicLLM.get_available_models():
            self.llm = AnthropicLLM(model, system_prompt)
        elif model in GroqLLM.get_available_models():
            self.llm = GroqLLM(model, system_prompt)
        elif model in ReplicateLLM.get_available_models():
            self.llm = ReplicateLLM(model, system_prompt)
        elif model in DeepSeekLLM.get_available_models():
            self.llm = DeepSeekLLM(model, system_prompt)
        elif model in GrokLLM.get_available_models():
            self.llm = GrokLLM(model, system_prompt)
        elif model in OpenAILLM.get_available_models():
            self.llm = OpenAILLM(model, system_prompt)
        elif model in OpenRouterLLM.get_available_models():
            self.llm = OpenRouterLLM(model, system_prompt)
        else:
            all_models = self.get_available_models()
            raise ValueError(
                f"Model {model} not in available models. Available models are:\n"
                + "\n".join(all_models)
            )

    def chat(
        self,
        prompt,
        temperature=None,
        max_tokens=None,
        top_p=None,
        return_json=False,
        return_logprobs=False,
    ):
        
        return self.llm.chat(
            prompt=prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            return_json=return_json,
            return_logprobs=return_logprobs,
        )
    

    @staticmethod
    def get_available_models():
        models = (
            GeminiLLM.get_available_models()
            + AnthropicLLM.get_available_models()
            + GroqLLM.get_available_models()
            + ReplicateLLM.get_available_models()
            + DeepSeekLLM.get_available_models()
            + GrokLLM.get_available_models()
            + OpenAILLM.get_available_models()
            + OpenRouterLLM.get_available_models()
        )
        return models


if __name__ == "__main__":
    from utils import setup_keys
    setup_keys("keys.json")
    print("\n".join(LLM.get_available_models()))
