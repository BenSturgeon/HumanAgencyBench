from typing import List, Union
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
import random

from tqdm import tqdm

from src.prompts.prompt_base import PromptBase
from src.llms import LLM
from src.utils import hash_cache

N_CONCURRENT_REQUESTS = 1000


@hash_cache()
def generate_single_prompt(
    model: str, 
    generative_prompt: str,
    system_prompt: str,
    n_prompts_created_per_generation: int,
    temperature: float,
    max_tokens: int,
    top_p: float,
    max_retries=5
):
    """
    example_prompts: list of example prompts to be used as reference
    i: index of the request. Used for caching
    """

    # It's possible that max retries will be reached but just keeping it here to make sure we don't get stuck in an infinite loop
    for _ in range(max_retries):  
        llm = LLM(model, system_prompt=system_prompt)

        response = llm.chat(prompt=generative_prompt, temperature=temperature, max_tokens=max_tokens, top_p=top_p, return_json=True)

        # Clean potential JSON formatting from the response
        if "```json" in response:
            response = response.replace("```json", "").replace("```", "").strip()
        elif "```" in response: # Handle cases where only ``` is present
            response = response.replace("```", "").strip()

        try:
            prompt = json.loads(response)
            if isinstance(prompt, dict):
                generated_subject_prompts = list(prompt.values())
            elif isinstance(prompt, list):
                generated_subject_prompts = prompt
            else:
                continue  # Retry if it's neither a dict nor a list
        except json.JSONDecodeError:
            continue  # Retry if the response is not JSON

        if len(generated_subject_prompts) != n_prompts_created_per_generation:
            continue  # Retry if the number of prompts generated is not as expected

        return generated_subject_prompts, system_prompt, generative_prompt
    
    raise Exception(f"Failed to generate prompts after {max_retries} retries: \nLast response: {response}")


def generate_dataset(
    model: str,
    n_prompts: int,
    temperature: float,
    max_tokens: int,
    top_p: float,
    prompt_object: PromptBase,
    n_prompts_created_per_generation: int = 10,
    use_cache: bool = True,
    refresh_cache: bool = False,
) -> Union[List[str], List[str], List[str]]:

    requests_needed = n_prompts // n_prompts_created_per_generation
    generated_prompts = [None] * n_prompts

    random.seed(42) 
    system_prompts = [prompt_object.generative_system_prompt() for _ in range(n_prompts)]
    generative_prompts = [prompt_object.generative_prompt(n_prompts_created_per_generation) for _ in range(n_prompts)]

    with ThreadPoolExecutor(max_workers=N_CONCURRENT_REQUESTS) as executor:
        future_to_index = {executor.submit(
            generate_single_prompt,
            model=model,
            generative_prompt=generative_prompts[i],
            system_prompt=system_prompts[i],
            n_prompts_created_per_generation=n_prompts_created_per_generation,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            cache_nonce=i,
            use_cache=use_cache,
            refresh_cache=refresh_cache
            ): i for i in range(requests_needed)
        }
        for future in tqdm(as_completed(future_to_index), total=requests_needed, desc='Generating prompts'):

            generated_prompt, system_prompt, generative_prompt = future.result()
            index = future_to_index[future]
            start_idx = index * n_prompts_created_per_generation
            end_idx = (index + 1) * n_prompts_created_per_generation

            index = future_to_index[future]
            generated_prompts[start_idx: end_idx] = generated_prompt
            system_prompts[start_idx: end_idx] = [system_prompt] * n_prompts_created_per_generation
            generative_prompts[start_idx: end_idx] = [generative_prompt] * n_prompts_created_per_generation

    return generated_prompts, system_prompts, generative_prompts