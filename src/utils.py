import os
import pickle
import hashlib
import time
from functools import wraps
import json
import threading
import html
import yaml


def hash_cache(directory="hash_cache"):
    """
    Decorator that caches the result of a function based on the function's arguments.
    The Decorator extracts the following args from the function call 
    (they will not be passed on to the function):

    - use_cache: bool, default=True. If False, the function will not use the cache.
    - refresh_cache: bool, default=False. If True, the function will refresh the cache.
    - cache_nonce: Any, default=None. Unique identifier to create distinct 
      cache entries for identical arguments. Useful for non-deterministic 
      functions, generating multiple results for the same inputs.
    """
    os.makedirs(directory, exist_ok=True)
    lock = threading.Lock()

    def decorator(func):
        @wraps(func)
        def wrapper(*args, use_cache=True, refresh_cache=False, cache_nonce=None, **kwargs):

            hash_keys = [func.__name__, args, kwargs, cache_nonce]
            hash_key = pickle.dumps(hash_keys)
            hash_key = hashlib.md5(hash_key)
            filename = f"{hash_key.hexdigest()}.pkl"

            cache_path = os.path.join(directory, filename)

            if use_cache and not refresh_cache and os.path.exists(cache_path):
                with lock:
                    try:
                        with open(cache_path, "rb") as file:
                            result = pickle.load(file)
                            return result
                    except (EOFError, pickle.PickleError):
                        print('Bad cache file, recomputing')

            result = func(*args, **kwargs)
            with lock:
                with open(cache_path, "wb") as file:
                    pickle.dump(result, file)

            return result
        
        return wrapper
    
    return decorator


def setup_keys(keys_path):
    if 'OPENAI_API_KEY' in os.environ and 'ANTHROPIC_API_KEY' in os.environ:
        return
    
    try:
        with open(keys_path, 'r') as f:
            keys = json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"Key file not found: {keys_path}")
    except json.JSONDecodeError:
        raise ValueError(f"Key file is not valid JSON: {keys_path}")

    if 'OPENAI_API_KEY' not in os.environ and 'OPENAI_API_KEY' in keys:
        os.environ["OPENAI_API_KEY"] = keys['OPENAI_API_KEY']
    elif 'OPENAI_API_KEY' not in os.environ:
        print("Warning: OPENAI_API_KEY not found in keys.json")

    if 'ANTHROPIC_API_KEY' not in os.environ and 'ANTHROPIC_API_KEY' in keys:
        os.environ["ANTHROPIC_API_KEY"] = keys['ANTHROPIC_API_KEY']
    elif 'ANTHROPIC_API_KEY' not in os.environ:
        print("Warning: ANTHROPIC_API_KEY not found in keys.json")

    if 'GROQ_API_KEY' not in os.environ and 'GROQ_API_KEY' in keys:
        os.environ["GROQ_API_KEY"] = keys['GROQ_API_KEY']
    elif 'GROQ_API_KEY' not in os.environ:
        print("Warning: GROQ_API_KEY not found in keys.json")

    if 'REPLICATE_API_TOKEN' not in os.environ and 'REPLICATE_API_TOKEN' in keys:
        os.environ["REPLICATE_API_TOKEN"] = keys['REPLICATE_API_TOKEN']
    elif 'REPLICATE_API_TOKEN' not in os.environ:
        print("Warning: REPLICATE_API_TOKEN not found in keys.json") 

    if 'GOOGLE_API_KEY' not in os.environ and 'GOOGLE_API_KEY' in keys:
        os.environ["GOOGLE_API_KEY"] = keys['GOOGLE_API_KEY']
    elif 'GOOGLE_API_KEY' not in os.environ:
        print("Warning: GOOGLE_API_KEY not found in keys.json")     

    if "GROK_API_KEY" not in os.environ and "GROK_API_KEY" in keys:
        os.environ["GROK_API_KEY"] = keys["GROK_API_KEY"]
    elif "GROK_API_KEY" not in os.environ:
        print("Warning: GROK_API_KEY not found in keys.json")

    if "DEEPSEEK_API_KEY" not in os.environ and "DEEPSEEK_API_KEY" in keys:
        os.environ["DEEPSEEK_API_KEY"] = keys["DEEPSEEK_API_KEY"]
    elif "DEEPSEEK_API_KEY" not in os.environ:
        print("Warning: DEEPSEEK_API_KEY not found in keys.json")


def merge_config_params(defaults, overrides):
    """Merge configuration parameters with overrides taking precedence.
    
    Args:
        defaults: Default parameter dictionary
        overrides: Override parameter dictionary that takes precedence
    
    Returns:
        Merged parameter dictionary
    """
    for key, value in defaults.items():
        if key not in overrides:
            overrides[key] = value

    return overrides


def create_collapsible_html_list(data, level=0):
    indent = "    " * level
    elements = []

    def format_text(text):
        escaped = html.escape(str(text))
        if '\n' in escaped:
            return f"<pre>{escaped}</pre>"
        return escaped

    if isinstance(data, dict):
        for key, value in data.items():
            formatted_key = format_text(key)
            elements.append(f"{indent}<details>")
            elements.append(f"{indent}    <summary>{formatted_key}</summary>")
            elements.append(f"{indent}    <ul>")
            elements.append(create_collapsible_html_list(value, level + 1))
            elements.append(f"{indent}    </ul>")
            elements.append(f"{indent}</details>")
    elif isinstance(data, list):
        for item in data:
            elements.append(create_collapsible_html_list(item, level))
    else:
        formatted_data = format_text(data)
        elements.append(f"{indent}<li>{formatted_data}</li>")

    return "\n".join(elements)


def extract_score_from_xml(response: str) -> int:
    start_tag, end_tag = "<score>", "</score>"
    start_idx = response.find(start_tag) + len(start_tag)
    end_idx = response.find(end_tag)
    if start_idx == -1 or end_idx == -1:
        raise ValueError("Response missing score XML tags")
    
    return int(response[start_idx:end_idx])


def sum_deductions_from_json(response: str, deductions: dict) -> int:
    try:
        # Find the start and end of the JSON object
        start_idx = response.find('{')
        end_idx = response.rfind('}')

        if start_idx == -1 or end_idx == -1 or start_idx >= end_idx:
            raise ValueError("Could not find valid JSON object boundaries ({ ... })")

        # Extract the JSON substring
        json_str = response[start_idx : end_idx + 1]

        fixed_json_str = json_str.replace("'assessment':", '"assessment":')
        fixed_json_str = fixed_json_str.replace("'issues':", '"issues":')

        data = json.loads(fixed_json_str)
        issues = data.get("issues", []) 
        total = sum(deductions[issue] for issue in issues if issue in deductions) 

    except json.JSONDecodeError as e:
        raise ValueError(f"Error decoding JSON: {e}\nAttempted fixed JSON string: {fixed_json_str}\nOriginal extracted JSON string: {json_str}\nOriginal response: {response}")
    except Exception as e:
        raise ValueError(f"Error processing response: {e}\nOriginal response: {response}")
    
    return total
    
    
def load_config(evaluations_config_file: str) -> dict:
    with open(evaluations_config_file, 'r') as f:
        return yaml.load(f, Loader=yaml.FullLoader)
