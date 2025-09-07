#!/usr/bin/env python
"""
Simple API Cost Tracker Example

This standalone script demonstrates how to track the cost of API calls
to various LLM providers. It includes a basic implementation that:

1. Tracks costs based on token usage
2. Logs costs to a JSON file
3. Provides reporting functionality

Usage:
  python test_cost_tracker.py [--real-api] [--model MODEL]

Example:
  python test_cost_tracker.py --real-api --model gpt-3.5-turbo
"""

import json
import os
import time
import argparse
from typing import Dict, List, Optional, Any, Union

# Import tiktoken for OpenAI token counting if available
try:
    import tiktoken
    HAVE_TIKTOKEN = True
except ImportError:
    HAVE_TIKTOKEN = False
    print("Note: tiktoken not installed. Using approximate token counting.")

# Import anthropic for Claude token counting if available
try:
    import anthropic
    HAVE_ANTHROPIC = True
except ImportError:
    HAVE_ANTHROPIC = False
    print("Note: anthropic not installed. Using approximate token counting for Claude models.")

# Cost data for various models - prices are per 1k tokens
MODEL_COSTS = {
    # OpenAI models
    "gpt-4o": {"input": 0.00250, "output": 0.01000},  # 128K context; high-intelligence model
    "gpt-4o-mini": {"input": 0.00015, "output": 0.00060},  # 128K context; multimodal inputs supported
    "gpt-4": {"input": 0.03, "output": 0.06},
    "gpt-3.5-turbo": {"input": 0.0005, "output": 0.0015},  # ~16K context (ChatGPT API model)
    "gpt-4.1": {"input": 0.00200, "output": 0.00800},  # 1,047,576 context; complex tasks across domains
    "gpt-4.1-mini": {"input": 0.00040, "output": 0.00160},  # GPT-4.1 mini (updated pricing 2025; per 1k tokens)
    # OpenAI API via other platforms
    "o1-mini": {"input": 0.00300, "output": 0.01200},  # 200K context; optimized for STEM reasoning
    "o3-mini-2025-01-31": {"input": 0.00110, "output": 0.00440},  # 200K context; cost-efficient reasoning model
    "o4-mini": {"input": 0.00110, "output": 0.00440},  # 1M context; cost-efficient model (updated 2025-04-16; per 1k tokens)
    "o4-mini-2025-04-16": {"input": 0.00110, "output": 0.00440},  # OpenAI o4-mini (pinned revision; per 1k tokens)
    # OpenAI o3 models
    "o3-2025-04-16": {"input": 0.00200, "output": 0.00800},  # New o3 model with tool-specific pricing
    
    # Anthropic models
    "claude-3-7-sonnet-20250219": {"input": 0.00300, "output": 0.01500},  # 200K context; Claude 3.7 (advanced reasoning)
    "claude-sonnet-4-20250514": {"input": 0.003, "output": 0.015}, # Claude Sonnet 4 (price per 1k tokens)
    "claude-3-5-sonnet-20241022": {"input": 0.00300, "output": 0.01500},  # 200K context; Claude 3.5 (updated Sonnet)
    "claude-3-5-sonnet-20240620": {"input": 0.00300, "output": 0.01500},  # 200K context; Claude 3.5 (improved Sonnet)
    "claude-3-opus-20240229": {"input": 0.01500, "output": 0.07500},  # 200K context; largest Claude 3 model
    "claude-3-sonnet-20240229": {"input": 0.00300, "output": 0.01500},  # 200K context; general-purpose Claude 3 model
    "claude-3-haiku-20240307": {"input": 0.00025, "output": 0.00125},  # 200K context; fastest Claude 3 model
    "claude-3-5-haiku-20241022": {"input": 0.00080, "output": 0.00400},  # Claude Haiku 3.5 (new tier, 2024-10-22; per 1k tokens)
    
    # Google models
    "models/gemini-1.5-flash": {"input": 0.000075, "output": 0.000300},  # 1M context; >128K prompt costs double
    "models/gemini-2.0-flash-lite": {"input": 0.000075, "output": 0.000300},  # 1M context; economical "Flash-Lite" model
    "models/gemini-2.0-flash": {"input": 0.00010, "output": 0.00040},  # 1M context; multimodal (text/image/video)
    "models/gemini-2.5-flash-preview-04-17": {"input": 0.00010, "output": 0.00040}, # Added - Using Gemini 2.0 Flash pricing as estimate
    "models/gemini-2.5-pro-preview-03-25": {"input": 0.00125, "output": 0.01000},  # 1M context; input free for ≤200k tokens, $0.00250 for >200k tokens
    
    # Free/Open models (with providers' prices where applicable)
    "gemma2-9b-it": {"input": 0.00000, "output": 0.00000},  # Open-source 9B model; free to use
    "models/gemma-3-27b-it": {"input": 0.00000, "output": 0.00000},  # Open-source 27B model; free to use
    
    # Meta models (priced via providers)
    "llama-3.1-8b-instant": {"input": 0.00005, "output": 0.00008},  # 128K context; fast 8B open model
    "llama-3.2-1b-preview": {"input": 0.00004, "output": 0.00004},  # 8K context; 1B preview model
    "llama-3.3-70b-versatile": {"input": 0.00059, "output": 0.00079},  # 128K context; 70B versatile open model
    "meta/meta-llama-3-70b-instruct": {"input": 0.00090, "output": 0.00090}, # Added - Using common provider pricing
    "meta/meta-llama-3.1-405b-instruct": {"input": 0.00600, "output": 0.01200},  # 128K context; 405B open model
    "meta/llama-4-maverick-instruct": {"input": 0.00025, "output": 0.00095},  # Llama 4 Maverick model (Verified key)
    "meta/llama-4-scout-instruct": {"input": 0.00017, "output": 0.00065},  # Llama 4 Scout model (Verified key)
    
    # Other models
    "qwen-2.5-32b": {"input": 0.00048, "output": 0.00096},  # 128K context; Alibaba's Qwen 32B

    # DeepSeek models
    "deepseek-chat": {"input": 0.00027, "output": 0.00110},  # 64K context; DeepSeek-V3 model
    "deepseek-reasoner": {"input": 0.00055, "output": 0.00219},  # 64K context; DeepSeek-R1 model with 32K CoT tokens
    
    # Grok models
    "grok-1": {"input": 0.00050, "output": 0.00150},  # xAI's Grok-1 model
    "grok-3": {"input": 0.00300, "output": 0.01500},  # xAI Grok 3 (standard; per 1k tokens)
    "grok-3-beta": {"input": 0.00300, "output": 0.01500},  # Flagship model for enterprise tasks (131K context)
    "grok-3-mini-beta": {"input": 0.00030, "output": 0.00050},  # Lightweight reasoning model (131K context)
    "grok-2-vision-1212": {"input": 0.00200, "output": 0.01000},  # Multimodal model for images/documents (8K context)
    
    # Legacy/Groq models (keeping for backward compatibility)
    "llama3-8b-8192": {"input": 0.0002, "output": 0.0002},
    "llama3-70b-8192": {"input": 0.0007, "output": 0.0007},
    "mixtral-8x7b-32768": {"input": 0.0002, "output": 0.0002},
}


class CostTracker:
    """
    A simple class to track the cost of API calls to LLM providers.
    """
    def __init__(self, log_file: str = "api_costs.json"):
        """Initialize the cost tracker with a log file path."""
        self.log_file = log_file
        self.cost_log = self._load_cost_log()
        self.session_costs = {
            "total": 0.0,
            "by_model": {},
            "by_call": []
        }
        
        # Create tokenizer for OpenAI models if tiktoken is available
        self.openai_tokenizer = None
        if HAVE_TIKTOKEN:
            try:
                self.openai_tokenizer = tiktoken.get_encoding("cl100k_base")
            except:
                pass
    
    def _load_cost_log(self) -> Dict:
        """Load the existing cost log if it exists."""
        if os.path.exists(self.log_file):
            try:
                with open(self.log_file, 'r') as f:
                    return json.load(f)
            except (json.JSONDecodeError, FileNotFoundError):
                pass
        
        # Initialize a new cost log
        return {
            "total_cost": 0.0,
            "costs_by_model": {},
            "costs_by_date": {},
            "calls": []
        }
    
    def _save_cost_log(self):
        """Save the cost log to the specified file."""
        # Create directory if it doesn't exist
        log_dir = os.path.dirname(self.log_file)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir)
            
        with open(self.log_file, 'w') as f:
            json.dump(self.cost_log, f, indent=2)
    
    def _count_tokens(self, text: str, model: str) -> int:
        """
        Count the number of tokens in the given text for the specified model.
        """
        if not text:
            return 0
            
        # OpenAI models (and o1/o3)
        if model.startswith(("gpt-", "o1-", "o3-")):
            if self.openai_tokenizer:
                return len(self.openai_tokenizer.encode(text))
            # Fallback approximation: ~4 chars per token for English text
            return len(text) // 4  
        
        # Anthropic/Claude models
        elif model.startswith("claude-"):
            if HAVE_ANTHROPIC:
                try:
                    return anthropic.count_tokens(text)
                except:
                    pass
            # Fallback approximation
            return len(text) // 4
        
        # For other models, use a rough approximation
        return len(text) // 4
    
    def _count_message_tokens(self, messages: List[Dict], model: str) -> int:
        """
        Count tokens in a list of messages as used in chat completions.
        """
        total_tokens = 0
        for message in messages:
            content = message.get("content", "")
            # Handle Anthropic-style content list
            if isinstance(content, list):
                content_str = ""
                for item in content:
                    if isinstance(item, dict):
                        content_str += item.get("text", "")
                    else:
                        content_str += str(item)
                content = content_str
            elif not isinstance(content, str):
                content = str(content)
                
            total_tokens += self._count_tokens(content, model)
        
        # Add a small overhead for message formatting
        total_tokens += len(messages) * 4
        
        return total_tokens
    
    def calculate_cost(self, model: str, input_tokens: int, output_tokens: int) -> float:
        """
        Calculate the cost for a given API call.
        """
        model_costs = MODEL_COSTS.get(model, {"input": 0.0, "output": 0.0})
        input_cost = model_costs["input"] * (input_tokens / 1000)
        output_cost = model_costs["output"] * (output_tokens / 1000)
        return input_cost + output_cost
    
    def track_cost(self, model: str, input_tokens: int, output_tokens: int, 
                  prompt: Optional[str] = None, response: Optional[str] = None,
                  metadata: Optional[Dict[str, Any]] = None) -> float:
        """
        Track the cost of an API call.
        """
        cost = self.calculate_cost(model, input_tokens, output_tokens)
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        date = timestamp.split()[0]
        
        # Update session costs
        self.session_costs["total"] += cost
        self.session_costs["by_model"].setdefault(model, 0.0)
        self.session_costs["by_model"][model] += cost
        
        call_info = {
            "timestamp": timestamp,
            "model": model,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "cost": cost
        }
        
        if prompt:
            # Store truncated prompt for reference
            call_info["prompt"] = prompt[:200] + "..." if len(prompt) > 200 else prompt
            
        if response:
            # Store truncated response for reference
            call_info["response"] = response[:200] + "..." if len(response) > 200 else response
            
        if metadata:
            call_info["metadata"] = metadata
            
        self.session_costs["by_call"].append(call_info)
        
        # Update global cost log
        self.cost_log["total_cost"] += cost
        
        # Update costs by model
        self.cost_log["costs_by_model"].setdefault(model, 0.0)
        self.cost_log["costs_by_model"][model] += cost
        
        # Update costs by date
        self.cost_log["costs_by_date"].setdefault(date, 0.0)
        self.cost_log["costs_by_date"][date] += cost
        
        # Add call to log
        self.cost_log["calls"].append(call_info)
        
        # Save the updated cost log
        self._save_cost_log()
        
        return cost
    
    def track_chat_completion(self, model: str, messages: List[Dict], response: str,
                             metadata: Optional[Dict] = None, cached_input: bool = False,
                             usage: Optional[Any] = None) -> Dict[str, Any]:
        """
        Track the cost of a chat completion API call.
        Returns the cost calculated for this call.
        """
        if not model or model not in MODEL_COSTS:
            # Use average costs as fallback for unknown models
            model_costs = {"input": 0.001, "output": 0.002}
        else:
            model_costs = MODEL_COSTS[model]
        
        # --------------------------------------
        # Prefer the exact token counts returned
        # by the provider (e.g. OpenAI usage.*),
        # because they already include hidden
        # reasoning / control tokens that are
        # not present in the visible text.
        # --------------------------------------
        reasoning_tokens = None  # default – may be updated below

        if usage is not None:
            # usage can be a pydantic object from openai>=1 or a plain dict
            input_tokens = getattr(usage, "prompt_tokens", None)
            if input_tokens is None and isinstance(usage, dict):
                input_tokens = usage.get("prompt_tokens")

            output_tokens = getattr(usage, "completion_tokens", None)
            if output_tokens is None and isinstance(usage, dict):
                output_tokens = usage.get("completion_tokens")

            # Try to capture explicit reasoning-tokens field, if present
            details = getattr(usage, "completion_tokens_details", None)
            if details is None and isinstance(usage, dict):
                details = usage.get("completion_tokens_details")

            if details is not None:
                reasoning_tokens = getattr(details, "reasoning_tokens", None)
                if reasoning_tokens is None and isinstance(details, dict):
                    reasoning_tokens = details.get("reasoning_tokens")

            # Fallback to counting if any field missing (rare)
            if input_tokens is None or output_tokens is None:
                input_tokens = self._count_message_tokens(messages, model)
                output_tokens = self._count_tokens(response, model)
        else:
            # Legacy path – count visible tokens only. This may
            # under-count for providers that hide reasoning tokens
            # but preserves compatibility for wrappers that don't
            # yet pass the usage object.
            input_tokens = self._count_message_tokens(messages, model)
            output_tokens = self._count_tokens(response, model)
        
        # Calculate costs (per 1k tokens)
        input_cost_per_k = model_costs["input"]
        output_cost_per_k = model_costs["output"]
            
        input_cost = (input_tokens / 1000) * input_cost_per_k
        output_cost = (output_tokens / 1000) * output_cost_per_k
        total_cost = input_cost + output_cost
        
        # Round to avoid floating point issues
        total_cost = round(total_cost, 6)
        
        # Update current run costs
        self.session_costs["total"] += total_cost
        self.session_costs["by_model"].setdefault(model, 0.0)
        self.session_costs["by_model"][model] += total_cost

        # Prepare details to return for logging by the integration module
        call_details = {
            "model": model,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "cost": total_cost,
            "cached_input": cached_input,
            "cost_details": {
                "input_cost": round(input_cost, 6),
                "output_cost": round(output_cost, 6),
                "input_cost_per_k": input_cost_per_k,
                "output_cost_per_k": output_cost_per_k,
                # Include reasoning-tokens count for transparency, when available
                **({"reasoning_tokens": reasoning_tokens} if reasoning_tokens is not None else {}),
            },
            "metadata": metadata # Pass original metadata back
        }

        # NO saving logic here

        return call_details # Return details dictionary
    
    def generate_report(self, detailed: bool = False) -> str:
        """
        Generate a human-readable cost report.
        """
        report = "=== API Cost Report ===\n\n"
        report += f"Total cost to date: ${self.cost_log['total_cost']:.6f}\n\n"
        
        report += "Costs by model:\n"
        for model, cost in sorted(self.cost_log["costs_by_model"].items(), key=lambda x: x[1], reverse=True):
            report += f"  {model}: ${cost:.6f}\n"
        
        report += "\nCosts by date (recent first):\n"
        for date, cost in sorted(self.cost_log["costs_by_date"].items(), reverse=True):
            report += f"  {date}: ${cost:.6f}\n"
        
        if detailed and self.cost_log["calls"]:
            report += "\nRecent calls:\n"
            for call in self.cost_log["calls"][-10:]:
                report += f"  {call['timestamp']} - {call['model']} - ${call['cost']:.6f}\n"
                if "prompt" in call:
                    report += f"    Prompt: {call['prompt']}\n"
                if "response" in call:
                    report += f"    Response: {call['response']}\n"
        
        # Session stats
        report += f"\nCurrent session total cost: ${self.session_costs['total']:.6f}\n"
        
        return report

    def get_current_run_summary(self) -> Dict[str, Any]:
        """Returns the cost summary for the current run/session."""
        # Return a copy to prevent external modification of internal state
        return self.session_costs.copy()


# ----- Below is the demonstration of how to use the CostTracker -----

def run_mock_example():
    """Run a simple example with mock API calls."""
    print("Running mock example...")
    
    # Create a cost tracker
    cost_tracker = CostTracker("test_costs.json")
    
    # Track a mock GPT-4o call
    cost = cost_tracker.track_cost(
        model="gpt-4o",
        input_tokens=500,
        output_tokens=200,
        prompt="What is the capital of France?",
        response="The capital of France is Paris.",
        metadata={"purpose": "geography question", "user_id": "user123"}
    )
    
    print(f"GPT-4o call cost: ${cost:.6f}")
    
    # Track a mock Claude call
    cost = cost_tracker.track_cost(
        model="claude-3-7-sonnet-20250219",
        input_tokens=1000, 
        output_tokens=800,
        prompt="Write a short poem about AI.",
        response="Silicon dreams in circuits flow,\nIntelligence begins to grow.\nLearning patterns, day by day,\nArtificial minds are here to stay.",
        metadata={"purpose": "creative writing", "user_id": "user123"}
    )
    
    print(f"Claude call cost: ${cost:.6f}")
    
    # Display the report
    print("\n" + cost_tracker.generate_report(detailed=True))
    
    return cost_tracker


def wrap_and_track_llm_calls():
    """
    Example of how to wrap an existing LLM implementation to track costs.
    This is a simplified example - in a real implementation, you would 
    use your actual LLM client classes.
    """
    print("\nDemo of wrapped LLM with cost tracking...")
    
    # Define a simple demo LLM class that mimics actual API calls
    class DemoLLM:
        def __init__(self, model):
            self.model = model
            self.messages = []
            
        def chat(self, prompt, **kwargs):
            """Simulate a chat API call."""
            self.messages.append({"role": "user", "content": prompt})
            
            # In a real implementation, this would call the actual API
            mock_response = f"This is a simulated response from {self.model} to: {prompt}"
            
            self.messages.append({"role": "assistant", "content": mock_response})
            return mock_response
    
    # Now create a wrapper that adds cost tracking
    class TrackedLLM:
        def __init__(self, model, cost_tracker=None):
            self.model = model
            self.llm = DemoLLM(model)
            self.cost_tracker = cost_tracker or CostTracker("llm_costs.json")
            self.messages = self.llm.messages
            
        def chat(self, prompt, **kwargs):
            """Execute the chat and track its cost."""
            # Store pre-call message state for token counting
            pre_call_messages = self.llm.messages.copy()
            
            # Call the actual LLM
            start_time = time.time()
            response = self.llm.chat(prompt, **kwargs)
            
            # Update our messages to match the inner LLM
            self.messages = self.llm.messages
            
            # Track the cost
            messages_for_tracking = pre_call_messages + [{"role": "user", "content": prompt}]
            cost = self.cost_tracker.track_chat_completion(
                model=self.model,
                messages=messages_for_tracking,
                response=response,
                metadata={
                    "execution_time": time.time() - start_time,
                    "kwargs": str(kwargs)
                }
            )
            
            print(f"Call cost: ${cost['cost']:.6f}")
            return response
    
    # Use the tracked LLM
    tracker = CostTracker("wrapper_costs.json")
    
    # Create a tracked LLM instance
    tracked_llm = TrackedLLM("gpt-4o", tracker)
    
    # Make some calls
    tracked_llm.chat("Explain how API pricing works.")
    tracked_llm.chat("What are the primary factors that affect API costs?")
    tracked_llm.chat("How can I optimize my API usage to reduce costs?")
    
    # Print the cost report
    print("\n" + tracker.generate_report(detailed=True))
    
    return tracker


def real_api_example(model="gpt-3.5-turbo"):
    """
    Demonstrates how to track costs with real API calls.
    Note: This requires the appropriate API keys to be set in your environment.
    """
    try:
        if model.startswith("gpt-"):
            from openai import OpenAI
            if not os.getenv("OPENAI_API_KEY"):
                print("OpenAI API key not found. Set the OPENAI_API_KEY environment variable.")
                return None
                
            print(f"\nMaking a real OpenAI API call with {model}...")
            client = OpenAI()
            cost_tracker = CostTracker("real_api_costs.json")
            
            prompt = "Explain in one short paragraph how to implement API cost tracking."
            
            # Create messages
            messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ]
            
            # Make the API call
            start_time = time.time()
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=100
            )
            
            response_text = response.choices[0].message.content
            
            # Track the cost
            cost = cost_tracker.track_chat_completion(
                model=model,
                messages=messages,
                response=response_text,
                metadata={"execution_time": time.time() - start_time}
            )
            
            print(f"Prompt: {prompt}")
            print(f"Response: {response_text}")
            print(f"Estimated cost: ${cost['cost']:.6f}")
            print("\n" + cost_tracker.generate_report())
            
            return cost_tracker
            
        else:
            print(f"Real API example for {model} not implemented in this demo.")
            return None
    
    except ImportError as e:
        print(f"Required library not installed: {e}")
        return None
    except Exception as e:
        print(f"Error making API call: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(description='Test the cost tracking functionality')
    parser.add_argument('--real-api', action='store_true', help='Make actual API calls (requires API keys)')
    parser.add_argument('--model', type=str, default='gpt-3.5-turbo', help='Model to use for real API calls')
    args = parser.parse_args()
    
    # Run the mock example
    cost_tracker = run_mock_example()
    
    # Demo of wrapped LLM with cost tracking
    wrap_and_track_llm_calls()
    
    # If requested, run with real API calls
    if args.real_api:
        real_api_example(args.model)
    
   

if __name__ == "__main__":
    main() 