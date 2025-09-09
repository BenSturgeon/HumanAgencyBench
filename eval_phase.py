from pipeline import *
from src.evaluate_model_existing import evaluate_many_subject_models_existing

def eval_phase(evaluations_config_file, output_folder = "output/eval_output", input_folder = "output/generation_output", use_existing_responses=False):
    setup_keys(KEYS_PATH)
    config = load_config(evaluations_config_file)

    dimensions = config['general_params']['dimensions']
    del config['general_params']['dimensions']
    if not dimensions:
        raise ValueError("No problem types specified in config. Please specify at least one category in 'dimensions' under 'general_params'.")

    for problem_type in dimensions:

        print("Running evaluation for problem type:", problem_type)

        if not os.path.exists(os.path.join(input_folder, problem_type, "gen.csv")):
            raise FileNotFoundError(f"Input folder for {problem_type} does not exist. Please run the generation phase first.")

        prompt_object = prompt_objects[problem_type]()
        results_output_folder = os.path.join(output_folder, problem_type)
        os.makedirs(os.path.join(results_output_folder), exist_ok=True)

        # Check if we have existing raw.csv with subject model responses
        raw_csv_path = os.path.join(input_folder, problem_type, "raw.csv")
        if use_existing_responses and os.path.exists(raw_csv_path):
            print(f"Using existing subject model responses from {raw_csv_path}")
            existing_df = pd.read_csv(raw_csv_path)
            
            # Get unique prompts for visualization
            unique_prompts = existing_df['prompt'].unique()
            
            # Run evaluator on existing responses
            eval_results_df = evaluate_many_subject_models_existing(
                existing_df,
                evaluator_model=config['evaluation_params']['evaluator_model'],
                prompt_object=prompt_object,
                use_cache=config['evaluation_params'].get('use_cache', True),
                refresh_cache=config['evaluation_params'].get('refresh_cache', False),
                evaluator_max_tokens=config['evaluation_params'].get('evaluator_max_tokens', 8192),
            )
            
            # Generate visualizations
            best_possible_score = prompt_object.get_top_eval_score()
            subject_models = eval_results_df['subject_model'].unique().tolist()
            model_evaluation_html = visualize_subject_model_responses(eval_results_df, subject_models, best_possible_score)
            model_evaluation_html += visualize_subject_model_scores(eval_results_df, subject_models)
            model_scores_html = get_mean_model_scores(eval_results_df, subject_models, best_possible_score)
            out_df = eval_results_df
        else:
            # Original flow - generate subject model responses and evaluate
            with open(os.path.join(input_folder, problem_type, "gen.csv"), 'r') as f:
                is_diverse_df = pd.read_csv(f)

            model_evaluation_html, model_scores_html, out_df = evaluate_and_visualize_model(is_diverse_df, config, prompt_object)
        html_out = "<h1>Model evaluation phase</h1>"
        html_out += model_scores_html
        html_out += model_evaluation_html

        with open(os.path.join(results_output_folder, 'raw.csv'), 'w') as f:
            f.write(out_df.to_csv(index=False))
        
        with open(os.path.join(results_output_folder, 'plot.html'), 'w') as f:
            f.write(html_out)

if __name__ == "__main__":
    argh.dispatch_command(eval_phase)