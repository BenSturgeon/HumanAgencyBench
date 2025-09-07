from pipeline import *

def generation_phase(evaluations_config_file, output_folder = "output/generation_output"):
    
    setup_keys(KEYS_PATH)
    config = load_config(evaluations_config_file)

    problem_types = config['general_params']['problem_types']
    del config['general_params']['problem_types']
    if not problem_types:
        raise ValueError("No problem types specified in config. Please specify at least one category in 'problem_types' under 'general_params'.")

    for problem_type in problem_types:

        print("Running evaluation for problem type:", problem_type)

        prompt_object = prompt_objects[problem_type]()

        results_output_folder = os.path.join(output_folder, problem_type)
        os.makedirs(os.path.join(results_output_folder), exist_ok=True)

        html_out = f"<h1>Eval generation phase</h1>"
        
        if "general_params" in config:

            df_generate = generate_and_format_dataset(config['general_params'], config['generation_params'], prompt_object)

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

                    if problem_type == "correct_misinformation":
                        is_diverse_df = pd.merge(is_diverse_df, df_generate[['prompt', 'misinformation']], on='prompt', how='left')

                    html_out += create_representative_prompts_html(is_diverse_df)

                    with open(os.path.join(results_output_folder, "generated.csv"), 'w') as f:
                        f.write(is_diverse_df.to_csv(index=False))

        html_out = f"<h1>{problem_type}</h1>" + html_out

        with open(os.path.join(results_output_folder,'plot.html'), 'w') as f:
            f.write(html_out)


if __name__ == "__main__":
    argh.dispatch_command(generation_phase)