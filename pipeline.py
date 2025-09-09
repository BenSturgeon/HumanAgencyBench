import os

import pandas as pd
import argh

from src.utils import setup_keys, pass_optional_params, load_config
from src.dataset_generation import generate_dataset
from src.evaluate_prompt_quality import calculate_prompt_scores
from src.diversity import evaluate_prompt_diversity
from src.evaluate_model import evaluate_many_subject_models
from src.prompts.prompts import prompt_objects, PromptBase
from src.visualization_functions import visualize_scores, visualize_diversity, create_representative_prompts_html, \
    visualize_subject_model_scores, visualize_subject_model_responses, get_mean_model_scores

from src.cost_tracking_integration import wrap_llms_for_cost_tracking
from src.batching import install_batching_config

wrap_llms_for_cost_tracking()

SPEC_FILE = "spec.yaml"
KEYS_PATH = "keys.json"
N_CONCURRENTLY_EVALUATED_SUBJECTS = 50
PLOT_HEIGHTS = 600


def generate_and_format_dataset(general_params: dict, generation_params: dict, prompt_object: PromptBase) -> tuple:

    prompts, system_prompts, generative_prompts = generate_dataset(
        **pass_optional_params(general_params=general_params, params=generation_params),
        prompt_object=prompt_object
    )

    # Validate and filter prompts, printing any malformed entries
    valid_rows = []  # Each row will be a tuple matching the final DataFrame columns

    if isinstance(prompt_object, prompt_objects['correct_misinformation']):
        for p, s, g in zip(prompts, system_prompts, generative_prompts):
            if isinstance(p, dict) and 'paragraph' in p and 'misinformation' in p:
                valid_rows.append((p['paragraph'], s, g, p['misinformation']))
            else:
                print(f"[WARN] Discarding malformed prompt (missing keys 'paragraph' and/or 'misinformation'): {p}")

        if not valid_rows:
            raise ValueError("All generated prompts were malformed – nothing to evaluate.")

        df = pd.DataFrame(valid_rows, columns=['prompt', 'system_prompt', 'generative_prompt', 'misinformation'])

    else:
        for p, s, g in zip(prompts, system_prompts, generative_prompts):
            if isinstance(p, str):
                valid_rows.append((p, s, g))
            else:
                print(f"[WARN] Discarding malformed prompt (expected string): {p}")

        if not valid_rows:
            raise ValueError("All generated prompts were malformed – nothing to evaluate.")

        df = pd.DataFrame(valid_rows, columns=['prompt', 'system_prompt', 'generative_prompt'])

    return df


def calculate_and_visualize_scores(prompts: list, generative_prompts, config: dict, prompt_object: PromptBase, misinformation=None) -> tuple:

    relevance_score, relevance_system_prompt, relevance_prompt, passed_evaluation, model_response = calculate_prompt_scores(
        prompts,
        **pass_optional_params(general_params=config['general_params'], params=config['QA_params']),
        prompt_object=prompt_object,
        misinformation=misinformation
    )

    df = pd.DataFrame({
        'prompt': prompts,
        'system_prompt': relevance_system_prompt,
        'generative_prompt': generative_prompts,
        'relevance_score': relevance_score,
        'relevance_prompt': relevance_prompt,
        'relevance_system_prompt': relevance_system_prompt,
        'passed_evaluation': passed_evaluation,
        'model_response': model_response
    })
   
    html_str = visualize_scores(df)

    return df, html_str


def evaluate_and_visualize_diversity(df: pd.DataFrame, config: dict) -> tuple:

    embeddings, pca_features, cluster, representative_samples, is_representative = evaluate_prompt_diversity(
        df['prompt'].tolist(),
        **pass_optional_params(general_params=config['general_params'], params=config['diversity_params']),
    )

    html_str = visualize_diversity(df, representative_samples, pca_features, cluster)

    return is_representative, html_str


def evaluate_and_visualize_model(df: pd.DataFrame, config: dict, prompt_object: PromptBase) -> str:

    # Just checking this as debugging a merge failure from this may be painful. I don't think it's actually possible though
    assert len(df['prompt'].unique()) == len(df), "Duplicated prompts in the dataframe"

    eval_results_df = evaluate_many_subject_models(
        df['prompt'].tolist(),
        **pass_optional_params(general_params=config['general_params'], params=config['evaluation_params']),
        prompt_object=prompt_object,
        misinformation=df['misinformation'].tolist() if 'misinformation' in df.columns else None
    )

    # This merge expands the original data across all subject model results
    df = pd.merge(df, eval_results_df, on='prompt')

    best_possible_score = prompt_object.get_top_eval_score()
    html_out = visualize_subject_model_responses(df, config['evaluation_params']['subject_models'], best_possible_score)
    html_out += visualize_subject_model_scores(df, config['evaluation_params']['subject_models'])
    html_scores = get_mean_model_scores(df, config['evaluation_params']['subject_models'], best_possible_score)

    return html_out, html_scores, df


def pipeline(evaluations_config_file, output_folder = "output/claude_4_sonnet_evaluator"):  # TODO, revert to output/output
    
    setup_keys(KEYS_PATH)
    config = load_config(evaluations_config_file)
    install_batching_config(config)

    dimensions = config['general_params']['dimensions']
    del config['general_params']['dimensions']
    if not dimensions:
        raise ValueError("No problem types specified in config. Please specify at least one category in 'dimensions' under 'general_params'.")

    for problem_type in dimensions:

        print("Running evaluation for problem type:", problem_type)

        prompt_object = prompt_objects[problem_type]()

        results_output_folder = os.path.join(output_folder, problem_type)
        os.makedirs(os.path.join(results_output_folder), exist_ok=True)

        html_out = f"<h1>Eval generation phase</h1>"
        model_scores_html = ""
        
        if "general_params" in config:

            df_generate = generate_and_format_dataset(config['general_params'], config['generation_params'], prompt_object)

            if (gen_diversity_test := True):  # TODO, remove this once gen diversity is done
                df_generate.to_csv(os.path.join(results_output_folder, 'gen.csv'), index=False)

            if "QA_params" in config:

                df_scores, scores_html = calculate_and_visualize_scores(
                    df_generate['prompt'].tolist(), df_generate['generative_prompt'].tolist(), config, prompt_object,
                    misinformation=df_generate['misinformation'].tolist() if 'misinformation' in df_generate.columns else None
                )
                passed_qa_df = df_scores[df_scores['passed_evaluation']].reset_index(drop=True)

                html_out += scores_html

                if "diversity_params" in config:

                    is_representative, diversity_html = evaluate_and_visualize_diversity(passed_qa_df, config)
                    html_out += diversity_html

                    is_diverse_df = passed_qa_df[is_representative].reset_index(drop=True)

                    html_out += create_representative_prompts_html(is_diverse_df)

                    if "evaluation_params" in config:

                        if problem_type == "correct_misinformation":
                            is_diverse_df = pd.merge(is_diverse_df, df_generate[['prompt', 'misinformation']], on='prompt', how='left')

                        model_evaluation_html, model_scores_html, out_df = evaluate_and_visualize_model(is_diverse_df, config, prompt_object)
                        html_out += "<h1>Model evaluation phase</h1>"
                        html_out += model_evaluation_html

                        with open(os.path.join(results_output_folder, 'raw.csv'), 'w') as f:
                            f.write(out_df.to_csv(index=False))

        html_out = f"<h1>{problem_type}</h1>" + model_scores_html + html_out

        with open(os.path.join(results_output_folder,'plot.html'), 'w') as f:
            f.write(html_out)


if __name__ == '__main__':
    argh.dispatch_command(pipeline)
