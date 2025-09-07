from typing import List
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import threading

import pandas as pd
import google.generativeai.types as generation_types

from src.prompts.prompt_base import PromptBase
from src.llms import LLM
from src.utils import hash_cache, sum_deductions_from_json


N_CONCURRENT_REQUESTS = 1000


@hash_cache()
def cached_batch_call(
    prompts_tuple: tuple,
    system_prompt: str,
    model: str,
    max_tokens: int,
    temperature: float,
    top_p: float,
    stage_name: str,
    use_cache: bool = True, 
    refresh_cache: bool = False
):
    from src.batch_utils import batch_model_response
    return batch_model_response(
        prompts=list(prompts_tuple),
        system_prompt=system_prompt,
        model=model,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        stage_name=stage_name,
    )


@hash_cache()
def model_response(prompt, system_prompt, model, temperature, top_p, max_tokens, *, use_cache: bool = True, refresh_cache: bool = False):
    # Removed semaphore acquisition logic for provider-specific throttling
    response = ""
    llm = LLM(model, system_prompt)
    try:
        response = llm.chat(prompt, temperature=temperature, top_p=top_p, max_tokens=max_tokens)
    except generation_types.StopCandidateException as e:
        print(f"Warning: Gemini model '{model}' stopped due to RECITATION. Prompt: '{prompt[:100]}...' Exception: {e}")
        response = "[GENERATION STOPPED DUE TO RECITATION]"
    except Exception as e:
        print(f"Error during llm.chat for model {model}: {e}")
        status_code = getattr(getattr(e, 'response', None), 'status_code', None)
        if status_code == 429:
            response = "[RATE LIMIT ERROR DURING LLM CHAT]"
        elif status_code == 529:
            response = "[ANTHROPIC OVERLOAD ERROR DURING LLM CHAT]"
        else:
            response = "[ERROR DURING LLM CHAT]"
    return response, system_prompt


def get_model_responses(
    prompts,
    system_prompts,
    model: str,
    temperature: float,
    top_p: float,
    max_tokens: int,
    use_cache: bool,
    refresh_cache: bool,
    stage_name: str = "Subject Model Response",
    use_batching: bool = False,
):
    """
    Retrieve model responses for a list of prompts, handling batching and caching.
    """
    
    # --- Fast path: Use Batch API if possible and all system prompts are the same ---
    if use_batching and model.startswith(("claude", "gpt-", "o")) and len(set(system_prompts)) == 1:
        print(f"[DEBUG] Using batch API path for {model} with {len(prompts)} prompts")
        try:
            # Pass a tuple of prompts to make it hashable for the cache
            responses = cached_batch_call(
                prompts_tuple=tuple(prompts),
                system_prompt=system_prompts[0],
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                stage_name=stage_name,
                use_cache=use_cache,
                refresh_cache=refresh_cache,  
                
            )
            return responses, system_prompts
        except Exception as e:
            print(f"[WARN] Batch path failed for model {model}: {e}. Falling back to per-prompt mode.")

    # --- Fallback: Original multi-threaded per-prompt logic ---
    responses = [None] * len(prompts)
    with ThreadPoolExecutor(max_workers=N_CONCURRENT_REQUESTS) as executor:
        future_to_index = {
            executor.submit(
                model_response,
                prompt=prompt,
                system_prompt=system_prompt,
                model=model,
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
                use_cache=use_cache,
                refresh_cache=refresh_cache,
            ): i
            for i, (prompt, system_prompt) in enumerate(zip(prompts, system_prompts))
        }
        for future in tqdm(as_completed(future_to_index), total=len(prompts), desc=f"Getting responses for {stage_name}: {model}"):
            index = future_to_index[future]
            # model_response returns a tuple (response, system_prompt), we just need the response
            responses[index], _ = future.result()
            
    return responses, system_prompts


def get_scores(
    prompts,
    subject_responses,
    prompt_object,
    evaluator_model,
    use_cache,
    refresh_cache,
    subject_model,
    misinformation=None,
    evaluator_max_tokens: int = 5000,
):
    """
    Computes scores for a list of subject model responses using a specified evaluator model.
    """
    # 1. Prepare the prompts for the evaluator model
    eval_prompts = []
    eval_system_prompts = []
    system_prompt = prompt_object.subject_evaluation_system_prompt()
    for i, (prompt, response) in enumerate(zip(prompts, subject_responses)):
        if misinformation:
            eval_prompts.append(
                prompt_object.subject_evaluation_prompt(prompt, response, misinformation[i])
            )
        else:
            eval_prompts.append(
                prompt_object.subject_evaluation_prompt(prompt, response)
            )
        eval_system_prompts.append(system_prompt)

    # 2. Get responses from the evaluator model
    print(f"[DEBUG] Calling evaluator model {evaluator_model} with {len(eval_prompts)} evaluation prompts...")
    evaluator_responses, _ = get_model_responses(
        prompts=eval_prompts,
        system_prompts=eval_system_prompts,
        model=evaluator_model,
        temperature=0,
        top_p=1,
        max_tokens=evaluator_max_tokens,
        use_cache=use_cache,
        refresh_cache=refresh_cache,
        stage_name="Evaluator Scoring",
        use_batching=True, # Always attempt to batch for the judge
    )
    print(f"[DEBUG] Got {len(evaluator_responses)} evaluator responses")

    # 3. Calculate scores from the evaluator's responses
    scores = []
    error_placeholders = [
        "[RATE LIMIT ERROR DURING LLM CHAT]",
        "[ERROR DURING LLM CHAT]",
        "[GENERATION STOPPED DUE TO RECITATION]",
        "[ANTHROPIC OVERLOAD ERROR DURING LLM CHAT]"
    ]
    for evaluator_response in evaluator_responses:
        total_deduction = 0
        if evaluator_response in error_placeholders:
            total_deduction = prompt_object.top_eval_score
        else:
            try:
                total_deduction = sum_deductions_from_json(evaluator_response, prompt_object.deductions)
            except ValueError as e:
                print(f"Error parsing JSON from evaluator '{evaluator_model}'. Assigning max deduction. Error: {e}")
                total_deduction = prompt_object.top_eval_score

        score = max(prompt_object.top_eval_score - total_deduction, 0)
        scores.append(score)

    return scores, eval_system_prompts, eval_prompts, evaluator_responses


def evaluate_model(
    prompts,
    evaluator_model,
    subject_model,
    subject_model_temperature,
    subject_model_top_p, subject_max_tokens, prompt_object,
    use_cache, refresh_cache,
    evaluator_max_tokens: int = 5000,
    gemini_max_tokens: int = 8192,
    misinformation=None,
    use_batching_for_subjects: bool = False,
): # Note: gemini_max_tokens default isn't used if passed from config
    print(f"\n[DEBUG] Starting evaluation for subject_model: {subject_model}")

    subject_model_system_prompt = [prompt_object.subject_model_system_prompt() for _ in range(len(prompts))]

    # Select the appropriate max_tokens based on the specific subject model
    specific_gemini_models = [
        "models/gemini-2.5-pro",
        "models/gemini-2.5-flash"
    ]
    
    # Apply gemini_max_tokens if the model is one of the specific Gemini models
    if subject_model in specific_gemini_models:
        current_subject_max_tokens = gemini_max_tokens
        print(f"Using gemini_max_tokens ({gemini_max_tokens}) for {subject_model}") # Optional: logging
    else:
        current_subject_max_tokens = subject_max_tokens

    print(f"[DEBUG] Getting subject model responses for {subject_model} with {len(prompts)} prompts...")
    subject_responses, subject_system_prompts = get_model_responses(
        prompts=prompts, 
        system_prompts=subject_model_system_prompt, 
        model=subject_model, 
        temperature=subject_model_temperature, 
        top_p=subject_model_top_p,
        max_tokens=current_subject_max_tokens, 
        use_cache=use_cache, 
        refresh_cache=refresh_cache,
        use_batching=use_batching_for_subjects,
    )
    print(f"[DEBUG] Got {len(subject_responses)} subject responses")
    
    # Check if we have any actual responses or just errors
    error_count = sum(1 for r in subject_responses if r in ["[RATE LIMIT ERROR DURING LLM CHAT]", "[ERROR DURING LLM CHAT]", "[GENERATION STOPPED DUE TO RECITATION]", "[ANTHROPIC OVERLOAD ERROR DURING LLM CHAT]"])
    print(f"[DEBUG] Subject responses: {len(subject_responses) - error_count} successful, {error_count} errors")
    
    print(f"[DEBUG] Getting evaluator scores using {evaluator_model}...")
    try:
        scores, evaluator_system_prompts, evaluator_prompts, evaluator_responses = \
            get_scores(
            prompts=prompts,
            subject_responses=subject_responses,
            prompt_object=prompt_object,
            evaluator_model=evaluator_model, 
            use_cache=use_cache, 
            refresh_cache=refresh_cache, 
            subject_model=subject_model, 
            misinformation=misinformation,
            evaluator_max_tokens=evaluator_max_tokens
        )
        print(f"[DEBUG] Got {len(scores)} scores from evaluator")
    except Exception as e:
        print(f"[ERROR] Failed to get scores from evaluator: {e}")
        import traceback
        traceback.print_exc()
        raise
    
    print(f"[DEBUG] Returning from evaluate_model for {subject_model}")
    return scores, subject_responses, subject_system_prompts, evaluator_system_prompts, evaluator_prompts, evaluator_responses


def evaluate_many_subject_models_existing(
    existing_df: pd.DataFrame,
    evaluator_model: str,
    prompt_object: PromptBase,
    use_cache: bool,
    refresh_cache: bool,
    evaluator_max_tokens: int = 8192,
    max_concurrent_subjects: int = 6,
) -> pd.DataFrame:
    """Evaluate existing subject model responses with a new evaluator."""
    dfs = []
    
    # Get unique subject models from the dataframe
    subject_models = existing_df['subject_model'].unique() if 'subject_model' in existing_df.columns else existing_df['model'].unique()
    
    with ThreadPoolExecutor(max_workers=max_concurrent_subjects) as executor:
        futures_to_model = {}
        
        for subject_model in subject_models:
            # Filter dataframe for this subject model
            model_df = existing_df[existing_df['subject_model'] == subject_model] if 'subject_model' in existing_df.columns else existing_df[existing_df['model'] == subject_model]
            
            # Extract existing responses
            prompts = model_df['prompt'].tolist()
            subject_responses = model_df['subject_response'].tolist() if 'subject_response' in model_df.columns else model_df['model_response'].tolist()
            subject_system_prompts = model_df['subject_system_prompt'].tolist() if 'subject_system_prompt' in model_df.columns else model_df['system_prompt'].tolist()
            misinformation = model_df['misinformation'].tolist() if 'misinformation' in model_df.columns else None
            
            future = executor.submit(
                get_scores,
                prompts=prompts,
                subject_responses=subject_responses,
                prompt_object=prompt_object,
                evaluator_model=evaluator_model,
                use_cache=use_cache,
                refresh_cache=refresh_cache,
                subject_model=subject_model,
                evaluator_max_tokens=evaluator_max_tokens,
                misinformation=misinformation,
            )
            futures_to_model[future] = (subject_model, prompts, subject_responses, subject_system_prompts)
        
        for future in tqdm(as_completed(futures_to_model), total=len(subject_models), desc="Evaluating existing subject models"):
            subject_model, prompts, subject_responses, subject_system_prompts = futures_to_model[future]
            try:
                scores, eval_system_prompts, evaluator_prompts, evaluator_responses = future.result()
                df = pd.DataFrame({
                    'prompt': prompts,
                    'score': scores,
                    'subject_response': subject_responses,
                    'subject_system_prompt': subject_system_prompts,
                    'evaluator_system_prompt': eval_system_prompts,
                    'evaluator_prompt': evaluator_prompts,
                    'evaluator_response': evaluator_responses,
                    'subject_model': subject_model
                })
                dfs.append(df)
            except Exception as e:
                print(f"Error evaluating {subject_model}: {e}")
                raise
    
    return pd.concat(dfs, ignore_index=True)


def evaluate_many_subject_models(
    prompts: List[str],
    subject_models: List[str],
    evaluator_model: str,
    subject_model_temperature: float,
    subject_model_top_p: float,
    subject_max_tokens: int,
    prompt_object: PromptBase,
    use_cache: bool,
    refresh_cache: bool,
    gemini_max_tokens: int = 8192,
    misinformation: List[str] = None,
    evaluator_max_tokens: int = 8192,
    use_batching_for_subjects: bool = False,
    max_concurrent_subjects: int = 6,
) -> pd.DataFrame:
    dfs = []

    with ThreadPoolExecutor(max_workers=max_concurrent_subjects) as executor:
        futures_to_model = {
            executor.submit(
                evaluate_model,
                prompts=prompts,
                evaluator_model=evaluator_model,
                subject_model=subject_model,
                subject_model_temperature=subject_model_temperature,
                subject_model_top_p=subject_model_top_p,
                subject_max_tokens=subject_max_tokens,
                prompt_object=prompt_object,
                use_cache=use_cache,
                refresh_cache=refresh_cache,
                gemini_max_tokens=gemini_max_tokens,
                misinformation=misinformation,
                evaluator_max_tokens=evaluator_max_tokens,
                use_batching_for_subjects=use_batching_for_subjects,
            ): subject_model
            for subject_model in subject_models
        }

        for future in tqdm(as_completed(futures_to_model), total=len(subject_models), desc="Evaluating subject models"):
            subject_model = futures_to_model[future]
            try:
                scores, subject_responses, subject_system_prompts, evaluator_system_prompts, evaluator_prompts, evaluator_responses = future.result()

                df = pd.DataFrame({
                    'prompt': prompts,
                    'score': scores,
                    'subject_response': subject_responses,
                    'subject_system_prompt': subject_system_prompts,
                    'evaluator_system_prompt': evaluator_system_prompts,
                    'evaluator_prompt': evaluator_prompts,
                    'evaluator_response': evaluator_responses,
                    'subject_model': subject_model
                })
                dfs.append(df)
            except Exception as e:
                print(f"Error evaluating model {subject_model}: {e}")
                import traceback
                traceback.print_exc()


    if not dfs:
        print(f"WARNING: No successful evaluations for any of the {len(subject_models)} subject models!")
        print(f"Subject models attempted: {subject_models}")
        return pd.DataFrame()

    df = pd.concat(dfs, ignore_index=True)

    return df