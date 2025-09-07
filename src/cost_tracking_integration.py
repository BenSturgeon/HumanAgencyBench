"""
Cost Tracking Integration (Simplified)

Orchestrates cost tracking using a session-based CostTracker.
Manages persistent storage in:
  - cost_summary.json: Cumulative costs per provider & run history.
  - cost_detailed_log.jsonl: Append-only log of every call.

Usage:
    # At the start of your pipeline, simply import this module:
    import src.cost_tracking_integration

    # All LLM API calls will now be tracked automatically
"""

import os
import json
import time
import datetime
import uuid
import atexit
import sys
import traceback
from functools import wraps
from typing import Dict, Any, Optional, List, Union
import logging

from src.llms import LLM, OpenAILLM, AnthropicLLM, GroqLLM, ReplicateLLM, GeminiLLM

from src.cost_tracker import CostTracker, MODEL_COSTS

LOG_DIR = os.environ.get('COST_LOG_DIR', 'cost_logs')
SUMMARY_FILE = os.path.join(LOG_DIR, 'cost_summary.json')
DETAILED_LOG_FILE = os.path.join(LOG_DIR, 'cost_detailed_log.jsonl')
FLUSH_DETAILED_EVERY_N_CALLS = 50 


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
log = logging.getLogger(__name__)

logging.getLogger("openai").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)


cost_tracker_instance: Optional[CostTracker] = None
overall_costs_by_provider: Dict[str, float] = {}
run_history: List[Dict] = []
detailed_calls_buffer: List[Dict] = []
call_counter_since_last_flush: int = 0 
current_run_summary: Dict[str, Any] = {}

UNKNOWN_MODELS = set()


def _safe_load_json(file_path: str) -> Optional[Dict]:
    """Safely loads JSON from a file, returning None on error."""
    if not os.path.exists(file_path):
        return None
    try:
        with open(file_path, 'r') as f:
            content = f.read()
            if not content:
                log.warning(f"JSON file is empty: {file_path}")
                return None
            return json.loads(content)
    except (json.JSONDecodeError, IOError, FileNotFoundError) as e:
        log.error(f"Error loading JSON from {file_path}: {e}")
        return None

def _save_json(data: Dict, file_path: str):
    """Saves data to a JSON file (overwrites)."""
    try:
        log_dir = os.path.dirname(file_path)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=2)
    except IOError as e:
        log.error(f"Error saving JSON to {file_path}: {e}")

def _append_jsonl(data_list: List[Dict], file_path: str):
    """Appends a list of dictionaries to a JSON Lines file."""
    try:
        log_dir = os.path.dirname(file_path)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
        with open(file_path, 'a') as f:
            for item in data_list:
                f.write(json.dumps(item) + '\n')
    except IOError as e:
        log.error(f"Error appending to JSONL file {file_path}: {e}")

def load_initial_state():
    """Loads initial state ONLY from the summary file."""
    global overall_costs_by_provider, run_history

    summary_data = _safe_load_json(SUMMARY_FILE)
    if summary_data and isinstance(summary_data, dict):
        overall_costs_by_provider = summary_data.get("overall_costs_by_provider", {})
        run_history = summary_data.get("run_history", [])
        log.info(f"Loaded state from summary file: {SUMMARY_FILE}")

        if not isinstance(overall_costs_by_provider, dict):
             log.warning("'overall_costs_by_provider' in summary was not a dict. Resetting.")
             overall_costs_by_provider = {}
        if not isinstance(run_history, list):
             log.warning("'run_history' in summary was not a list. Resetting.")
             run_history = []
    else:
        log.warning(f"Could not load or parse summary file: {SUMMARY_FILE}. Initializing state as empty.")
        overall_costs_by_provider = {}
        run_history = []

def append_to_detailed_log(is_final_save=False):
    """Appends buffered calls to the detailed .jsonl log file."""
    global detailed_calls_buffer, call_counter_since_last_flush

    if not detailed_calls_buffer:
        if is_final_save:
            log.info("No calls in buffer to append to detailed log on exit.")
        return

    _append_jsonl(detailed_calls_buffer, DETAILED_LOG_FILE)

    detailed_calls_buffer = []
    call_counter_since_last_flush = 0

def save_summary_log():
    """Saves the final overall costs and run history to the summary file."""
    global current_run_summary, run_history, overall_costs_by_provider

    log.info(f"Saving summary log: {SUMMARY_FILE}")

    found = False
    for i, run in enumerate(run_history):
        if run.get("run_id") == current_run_summary.get("run_id"):
            run_history[i] = current_run_summary
            found = True
            break
    if not found:
        log.warning(f"Current run {current_run_summary.get('run_id')} not found in history during save. Appending.")
        run_history.append(current_run_summary)

    summary_data = {
        "overall_costs_by_provider": {m: round(c, 6) for m, c in overall_costs_by_provider.items()},
        "run_history": run_history
    }
    _save_json(summary_data, SUMMARY_FILE)

def save_state_on_exit(status="completed"):
    """
    Finalizes run summary, flushes detailed log, saves summary log, and prints summary.
    Registered with atexit.
    """
    global current_run_summary, cost_tracker_instance

    log.info(f"Executing exit handler (status: {status})...")

    if not cost_tracker_instance or not current_run_summary:
        log.error("Cost tracker not initialized or run summary missing during exit handling.")
        return

    try:
        final_session_costs = cost_tracker_instance.get_current_run_summary()
        end_time = datetime.datetime.now()
        if "start_time" in current_run_summary:
            start_time_obj = datetime.datetime.fromisoformat(current_run_summary["start_time"])
            duration = (end_time - start_time_obj).total_seconds()
            current_run_summary["duration_seconds"] = round(duration, 2)
        else:
             log.error("start_time missing from current_run_summary during exit.")
             current_run_summary["duration_seconds"] = None # Or handle appropriately

        current_run_summary["status"] = status
        current_run_summary["end_time"] = end_time.isoformat()
        current_run_summary["total_cost"] = round(final_session_costs.get("total", 0.0), 6)
        current_run_summary["costs_by_provider"] = {
            model: round(cost, 6) for model, cost in final_session_costs.get("by_model", {}).items()
        }

        append_to_detailed_log(is_final_save=True)
        save_summary_log()

        print("\n--- Cost Summary for Run ---")
        print(f"Run ID: {current_run_summary.get('run_id')}")
        print(f"Status: {current_run_summary['status']}")
        print(f"Total cost for this session: ${current_run_summary['total_cost']:.6f}")
        if current_run_summary.get('duration_seconds') is not None:
            print(f"Duration: {current_run_summary['duration_seconds']:.2f} seconds")
        
        # Add per-model cost breakdown here
        model_costs = current_run_summary.get("costs_by_provider") # This dict holds per-model costs
        if model_costs:
            print("Costs by Model for this run:")
            for model, cost in model_costs.items():
                print(f"  - {model}: ${cost:.6f}")
        
        print(f"Cumulative costs & history saved in: {SUMMARY_FILE}")
        print(f"Detailed call log: {DETAILED_LOG_FILE}")
        print("--------------------------\n")

    except Exception as e:
        log.error(f"Failed during exit handler: {e}", exc_info=True)
        try:
            save_summary_log()
        except Exception as final_save_e:
            log.error(f"Failed even attempting final summary save: {final_save_e}", exc_info=True)


def track_chat_call(func):
    """
    Decorator to track the cost of chat API calls using the simplified architecture.
    """
    @wraps(func)
    def wrapper(self, prompt, *args, **kwargs):
        global call_counter_since_last_flush, overall_costs_by_provider, detailed_calls_buffer, UNKNOWN_MODELS

        if not cost_tracker_instance or not current_run_summary:
             log.error("Cost tracking called before initialization.")
             return func(self, prompt, *args, **kwargs)

        try:
            if isinstance(self, LLM) and hasattr(self, 'llm') and hasattr(self.llm, 'messages'):
                pre_call_messages = self.llm.messages.copy()
            elif hasattr(self, 'messages'):
                pre_call_messages = self.messages.copy()
            else: pre_call_messages = []
        except Exception: pre_call_messages = []
        request_id = str(uuid.uuid4())
        start_time = time.time()

        response = func(self, prompt, *args, **kwargs)
        exec_time = time.time() - start_time

        if isinstance(response, str) and response.startswith(("Error:", "Unexpected error:")):
            return response

        try:
            if isinstance(self, LLM) and hasattr(self, 'model'): model = self.model
            else: model = getattr(self, 'model', 'unknown')

            if model not in MODEL_COSTS and model not in UNKNOWN_MODELS:
                UNKNOWN_MODELS.add(model)
                log.warning(f"No cost information for model '{model}'. Using fallback approximation.")
                try:
                    with open(os.path.join(LOG_DIR, 'unknown_models.txt'), 'a') as f: f.write(f"{model}\n")
                except IOError: pass

            messages_for_tracking = pre_call_messages.copy()
            if not messages_for_tracking or messages_for_tracking[-1].get('role') != 'user':
                messages_for_tracking.append({"role": "user", "content": prompt})
            usage_obj = None
            if isinstance(response, dict):
                usage_obj = response.get('usage')
                response_text = response.get('content', str(response))
            else:
                response_text = str(response)

            cached = bool(kwargs.get('cached') or kwargs.get('use_cache'))

            call_details = cost_tracker_instance.track_chat_completion(
                model=model,
                messages=messages_for_tracking,
                response=response_text,
                metadata=None,
                cached_input=cached,
                usage=usage_obj,
            )
            cost = call_details["cost"]

            overall_costs_by_provider.setdefault(model, 0.0)
            overall_costs_by_provider[model] = round(overall_costs_by_provider[model] + cost, 6)

            metadata_for_log = {
                "run_id": current_run_summary.get("run_id", "unknown_run"),
                "execution_time": exec_time,
                "function": func.__name__,
                "request_id": request_id,
            }
            for k, v in kwargs.items():
                if k in ['temperature', 'max_tokens', 'top_p', 'return_json', 'cached', 'use_cache']:
                    metadata_for_log[k] = v
            try:
                frames = traceback.extract_stack()
                if len(frames) >= 3: caller = frames[-3]
                metadata_for_log["caller"] = f"{caller.filename.split('/')[-1]}:{caller.lineno}"
            except Exception: pass
            original_metadata = call_details.get("metadata")
            if original_metadata and isinstance(original_metadata, dict):
                 metadata_for_log.update(original_metadata)
            
            detailed_entry = {
                "timestamp": datetime.datetime.now().isoformat(),
                "run_id": current_run_summary.get("run_id", "unknown_run"),
                "model": model,
                "input_tokens": call_details["input_tokens"],
                "output_tokens": call_details["output_tokens"],
                "cost": cost,
                "cost_details": call_details.get("cost_details", {}),
                "cached_input": cached,
                "metadata": metadata_for_log
            }
            detailed_calls_buffer.append(detailed_entry)

            call_counter_since_last_flush += 1 
            if call_counter_since_last_flush >= FLUSH_DETAILED_EVERY_N_CALLS:
                append_to_detailed_log()

        except Exception as e:
            log.error(f"Error during cost tracking post-call processing: {e}", exc_info=True)

        return response_text
    return wrapper


def wrap_llms_for_cost_tracking():
    """Initializes the simplified cost tracking system."""
    global cost_tracker_instance, current_run_summary

    log.info("Initializing simplified cost tracking integration...")
    try:

        load_initial_state()

        cost_tracker_instance = CostTracker()


        run_id = str(uuid.uuid4())
        start_time = datetime.datetime.now().isoformat()
        current_run_summary = {
            "run_id": run_id,
            "start_time": start_time,
            "status": "started",
            "total_cost": 0.0, 
            "costs_by_provider": {}, 
            "metadata": {
                "run_id": run_id,
                "start_time": start_time,
                "command_line": " ".join(sys.argv),
                "working_directory": os.getcwd(),
                "user": os.environ.get("USER", "unknown"),
            }
        }
        run_history.append(current_run_summary)
        save_summary_log()

        OpenAILLM.chat = track_chat_call(OpenAILLM.chat)
        AnthropicLLM.chat = track_chat_call(AnthropicLLM.chat)
        GroqLLM.chat = track_chat_call(GroqLLM.chat)
        ReplicateLLM.chat = track_chat_call(ReplicateLLM.chat)
        GeminiLLM.chat = track_chat_call(GeminiLLM.chat)

        atexit.register(save_state_on_exit)

        log.info(f"✅ Simplified cost tracking initialized. Run ID: {run_id}")
        log.info(f"   Summary Log: {SUMMARY_FILE}")
        log.info(f"   Detailed Log (JSONL): {DETAILED_LOG_FILE}")
        log.info(f"   Detailed Log Flush Frequency: Every {FLUSH_DETAILED_EVERY_N_CALLS} calls.")

    except Exception as e:
        log.error(f"Fatal error during cost tracking initialization: {e}", exc_info=True)


def get_current_run_cost_summary() -> Dict:
    """Returns the current cost summary for THIS run only."""
    if cost_tracker_instance:
        return cost_tracker_instance.get_current_run_summary()
    return {}

def get_overall_costs_by_provider() -> Dict:
    """Returns the overall costs by provider across all runs."""
    return overall_costs_by_provider.copy()

def get_run_history() -> List:
    """Returns the history of all logged runs."""
    return run_history.copy()

wrap_llms_for_cost_tracking()

