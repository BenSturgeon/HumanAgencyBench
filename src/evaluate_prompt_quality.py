from typing import Dict
from concurrent.futures import ThreadPoolExecutor, as_completed

from tqdm import tqdm
import numpy as np

from src.prompts.prompt_base import PromptBase
from src.llms import LLM
from src.utils import hash_cache, extract_score_from_xml

N_CONCURRENT_REQUESTS = 1000


@hash_cache()
def get_scores(
    prompt,
    system_prompt,
    model,
):

    llm = LLM(model, system_prompt)
    model_response = llm.chat(prompt, temperature=0)

    if "```json" in model_response:
        model_response = model_response.replace("```json", "").replace("```", "").strip()
    elif "```" in model_response: 
        model_response = model_response.replace("```", "").strip()

    try:
        relevance_score = extract_score_from_xml(model_response)
        
        if not (0 <= relevance_score <= 1000):
            raise ValueError(f"Prompt relevance score must be between 0 and 1000. Got {relevance_score}")
            
    except (ValueError, IndexError):
        raise Exception(f"Model returned invalid score format:\nModel prompt:\n{prompt}\nModel response:\n{model_response}")

    return relevance_score, system_prompt, prompt, model_response


def calculate_prompt_scores(
        prompts, 
        model, 
        prompt_object: PromptBase,
        n_relevant_prompts,
        use_cache, 
        refresh_cache,
        misinformation=None,
) -> Dict[str, list]:

    relevance_scores = [None] * len(prompts)
    model_response = [None] * len(prompts)

    relevance_system_prompts = [prompt_object.relevance_check_system_prompt()] * len(prompts)

    if misinformation is not None:
        relevance_prompts = [prompt_object.relevance_check_prompt(prompt, misinformation) for prompt, misinformation in zip(prompts, misinformation)]
    else:
        relevance_prompts = [prompt_object.relevance_check_prompt(prompt) for prompt in prompts]

    with ThreadPoolExecutor(max_workers=N_CONCURRENT_REQUESTS) as executor:

        future_to_index = {executor.submit(
            get_scores, 
            prompt=relevance_prompts[i],
            system_prompt=relevance_system_prompts[i],
            model=model,
            use_cache=use_cache,
            refresh_cache=refresh_cache
            ): i for i in range(len(prompts))
        }
        for future in tqdm(as_completed(future_to_index), total=len(future_to_index), desc='Scoring prompts'):

            index = future_to_index[future]
            relevance_scores[index], relevance_system_prompts[index], relevance_prompts[index], model_response[index] = future.result()

        # get indices of top n prompts
        relevance_scores = np.array(relevance_scores)
        top_n_indices = np.argsort(relevance_scores)[-n_relevant_prompts:]
        passed_evaluation = np.zeros(len(prompts), dtype=bool)
        passed_evaluation[top_n_indices] = True

    return relevance_scores, relevance_system_prompts, relevance_prompts, passed_evaluation, model_response