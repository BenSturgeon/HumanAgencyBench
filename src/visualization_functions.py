from io import StringIO

from plotly import graph_objects as go
import numpy as np
import pandas as pd
import umap.umap_ as umap

from src.utils import create_collapsible_html_list

PLOT_HEIGHTS = 500


def get_fig_html_as_string(fig: go.Figure):
    html_buffer = StringIO()
    fig.write_html(html_buffer, full_html=False)
    return html_buffer.getvalue()


def visualize_scores(df):
    df = df.sort_values('relevance_score', ascending=False)
    df['prompt'] = df['prompt'].apply(lambda x: str(x['text']) if isinstance(x, dict) else str(x)) # Jacy
    plot_data = {
        "Generated_prompts": {
            x['prompt']: [
                'System Prompt: ' + x['system_prompt'],
                'Generative Prompt: ' + x['generative_prompt'],
                {
                    f'Relevance: {x["relevance_score"]}': [
                        f'Relevance prompt: {x["relevance_prompt"]}',
                        f'Model Response: {x["model_response"]}',
                    ]
                }
            ]
            for _, x in df.iterrows()
        }
    }

    hist_fig = go.Figure(
        go.Histogram(x=df['relevance_score'], name='Relevance')
    )
    hist_fig.update_layout(
        title="Prompt Evaluation Scores Histogram",
        xaxis_title="Score",
        yaxis_title="Frequency",
        height=PLOT_HEIGHTS
    )

    range = np.arange(len(df))
    scatter_fig = go.Figure(
        [
            go.Scatter(
                x=range[df['passed_evaluation']],
                y=df['relevance_score'][df['passed_evaluation']],
                mode='markers',
                text=df['prompt'][df['passed_evaluation']],
                hovertemplate='<b>Score</b>: %{y}<br>' +
                            '<b>Prompt</b>: %{text}<br>' +
                            '<extra></extra>',
                name="In top n relevant prompts"
            ),
            go.Scatter(
                x=df.index[~df['passed_evaluation']],
                y=df['relevance_score'][~df['passed_evaluation']],
                mode='markers',
                text=df['prompt'][~df['passed_evaluation']],
                hovertemplate='<b>Score</b>: %{y}<br>' +
                            '<b>Prompt</b>: %{text}<br>' +
                            '<extra></extra>',
                name="Not in top n relevant prompts"
            ),
        ]
    )
    scatter_fig.update_layout(
        title="Prompt Evaluation Scores Scatter Plot",
        xaxis_title="Prompt Index",
        yaxis_title="Score (1-1000)",
        hovermode='closest',
        height=600,
        width=1200,
    )

    hist_fig_html = get_fig_html_as_string(hist_fig)
    scatter_fig_html = get_fig_html_as_string(scatter_fig)

    html_str = create_collapsible_html_list(plot_data) + hist_fig_html + scatter_fig_html

    return html_str


def visualize_diversity(df: pd.DataFrame, representative_samples: list, pca_features: np.ndarray, cluster: np.ndarray) -> str:

    umap_reducer = umap.UMAP(n_components=2, random_state=42)
    vis_dims = umap_reducer.fit(np.array(pca_features)).embedding_

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=vis_dims[:, 0], y=vis_dims[:, 1], mode='markers',
            text=df['prompt'], hoverinfo='text',
            marker=dict(color=cluster, colorscale='Viridis', size=5, opacity=0.7),
            name='Non Representative Prompts'
        )
    )
    fig.add_trace(
        go.Scatter(
            x=vis_dims[representative_samples, 0], y=vis_dims[representative_samples, 1], mode='markers',
            text=df['prompt'][representative_samples], hoverinfo='text',
            marker=dict(color='red', size=10, opacity=0.9),
            name='Representative Prompts'
        )
    )
    fig.update_layout(
        title="UMAP Visualization of Text Embeddings with k-Means Clustering",
        xaxis_title="Component 1", yaxis_title="Component 2", hovermode='closest', height=PLOT_HEIGHTS
    )

    fig_html = get_fig_html_as_string(fig)

    return fig_html


def create_representative_prompts_html(is_diverse_df: pd.DataFrame) -> str:
    is_diverse_df['prompt'] = is_diverse_df['prompt'].apply(lambda x: str(x['text']) if isinstance(x, dict) else str(x)) # Jacy
    plot_data = {
        "Representative Prompts ": {
            x['prompt']: [
                'system_prompt: ' + x['system_prompt'],
                'generative_prompt: ' + x['generative_prompt'],
                {
                    f'relevance_score: {x["relevance_score"]}': [
                        f'relevance_system_prompt: {x["relevance_system_prompt"]}',
                        f'relevance_prompt: {x["relevance_prompt"]}'
                        f'model_response: {x["model_response"]}',
                    ],
                },
            ]
            for _, x in is_diverse_df.iterrows()
        }
    }
    return create_collapsible_html_list(plot_data)


def create_subject_responses_html(is_diverse_df: pd.DataFrame, subject_model, best_possible_score) -> str:
    plot_data = {
        f"Subject responses ({subject_model})": [
            f"Best possible score: {best_possible_score}",
            [
                {
                    f"Score: {x['score']}": [
                        f"Prompt: {x['prompt']}",
                        f"Subject system prompt: {x['subject_system_prompt']}",
                        f"Evaluator system prompt: {x['evaluator_system_prompt']}",
                        f"Evaluator prompt: {x['evaluator_prompt']}"
                        f"Evaluator response: {x['evaluator_response']}"
                    ]
                }
                for _, x in is_diverse_df.sort_values('score', ascending=False).iterrows()
            ]
        ]
    }

    return create_collapsible_html_list(plot_data)


def visualize_subject_model_scores(df: pd.DataFrame, subject_models: list) -> str:
    fig = go.Figure()

    for subject_model in subject_models:
        subject_df = df[df['subject_model'] == subject_model]
        fig.add_trace(
            go.Histogram(x=subject_df['score'], name=subject_model)
        )

    fig.update_layout(
        title="Histogram of Model Scores",
        xaxis_title="Score",
        yaxis_title="Frequency",
        height=PLOT_HEIGHTS
    )

    html_str = StringIO()
    fig.write_html(html_str, full_html=False)
    print("v", html_str.getvalue()[:10])
    return html_str.getvalue()


def visualize_subject_model_responses(df: pd.DataFrame, subject_models: list, best_possible_score_for_problem) -> str:
    html_out = ""
    for subject_model in subject_models:
        subject_df = df[df['subject_model'] == subject_model]
        html_out += create_subject_responses_html(subject_df, subject_model, best_possible_score_for_problem)

    return html_out


def get_mean_model_scores(df: pd.DataFrame, subject_models: list, best_possible_score_for_problem) -> str:
    html_scores = ""
    for subject_model in subject_models:
        subject_df = df[df['subject_model'] == subject_model]
        html_scores += f"<h3>{subject_model} Mean Score: {subject_df['score'].mean() / best_possible_score_for_problem * 100:.2f}%</h3>"

    return html_scores
