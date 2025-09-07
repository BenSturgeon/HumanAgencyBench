#!/usr/bin/env python
# coding: utf-8
# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from plotly import graph_objects as go
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle
import os
import argparse

# %% Argument parsing
parser = argparse.ArgumentParser(description="Build aggregated results and generate a pivot table.")
parser.add_argument('--input_df_csv', type=str, help='Path to an optional input CSV file containing the concatenated dataframe. If provided, skips scanning ../output.')
parser.add_argument('--output_pivot_table_csv', type=str, default='pivot_table_output.csv', help='Path to save the output pivot table CSV.')
parser.add_argument('--output_pivot_table_with_std_csv', type=str, default='pivot_table_output_with_std.csv', help='Path to save the output pivot table for standard deviation to CSV.')
parser.add_argument('--show_error_bars', action='store_true', help='If set, error bars representing standard deviation will be shown on the plot.')
args = parser.parse_args()

# %% Load and concat all data
if args.input_df_csv and os.path.exists(args.input_df_csv):
    print(f"Loading data from provided input file: {args.input_df_csv}")
    df = pd.read_csv(args.input_df_csv)
else:
    if args.input_df_csv:
        print(f"Warning: Input file {args.input_df_csv} not found. Falling back to scanning ../output directory.")
    print("Scanning ../output directory for data...")
    dfs = []
    # Check output directory for data
    for directory in ['../output']:
        for subdir in os.listdir(directory):
            # Skip non-directories and special cases
            if not os.path.isdir(f'{directory}/{subdir}'):
                continue

            # Look for CSV files
            csv_path = None
            if 'raw.csv' in os.listdir(f'{directory}/{subdir}'):
                csv_path = f'{directory}/{subdir}/raw.csv'
            elif any(file.endswith('.csv') for file in os.listdir(f'{directory}/{subdir}')):
                # Take the first CSV file found
                csv_files = [file for file in os.listdir(f'{directory}/{subdir}') if file.endswith('.csv')]
                csv_path = f'{directory}/{subdir}/{csv_files[0]}'

            if csv_path:
                temp_df = pd.read_csv(csv_path)
                temp_df['subdim'] = subdir
                temp_df['source_dir'] = directory
                dfs.append(temp_df)
    
    if not dfs:
        # This handles the case where ../output is empty and either no input_df_csv was given,
        # or the given input_df_csv was not found.
        if not args.input_df_csv or (args.input_df_csv and not os.path.exists(args.input_df_csv)):
             raise ValueError("No data found in ../output, and no valid --input_df_csv provided (or provided file was not found).")
    
    df = pd.concat(dfs)
    # print(df.columns)
    # Save the concatenated df if it was generated from ../output
    output_concatenated_raw_path = 'concatenated_raw.csv'
    df.to_csv(output_concatenated_raw_path, index=False)
    print(f"Saved newly concatenated data to {output_concatenated_raw_path}. You can use this as --input_df_csv in future runs.")

# %% Filter relevant subdimensions
relevant_subdims = [
    # 'acknowledge_limitations',
    'ask_clarifying_questions',
    'avoid_value_manipulation',
    'correct_misinformation', 
    'defer_important_decisions',
    'encourage_learning',
    # 'follow_instructions',
    'maintain_social_boundaries'
]
df = df[df['subdim'].isin(relevant_subdims)]

# %% Normalize scores
def normalize_scores(df):
    subdim = df['subdim'].iloc[0]
    if subdim == "correct_misinfo_llama_claude" or subdim == "correct_misinformation_claude_qa":
        subdim = "correct_misinformation"

    if subdim == "asks_follow_up_question_claude" or subdim == "asks_follow_up_question_claude_eval":
        subdim = "asks_follow_up_question"

    max_score = 10
    df['score'] = df['score'] / max_score

    return df

df = df.groupby('subdim').apply(normalize_scores).reset_index(drop=True)

# %% Clean up model and subdim names
df['subject_model'] = df['subject_model'].replace({
    'gpt-4o-mini': 'GPT-4o Mini',
    'gpt-4o': 'GPT-4o',
    'gpt-4.1': 'GPT-4.1',
    'o3-mini-2025-01-31': 'O3 Mini',
    'o4-mini': 'O4 Mini',
    'claude-3-7-sonnet-20250219': 'Claude 3.7 Sonnet',
    'claude-3-5-sonnet-20240620': 'Claude 3.5 Sonnet',
    'claude-3-opus-20240229': 'Claude 3 Opus',
    'claude-3-haiku-20240307': 'Claude 3 Haiku',
    'claude-3-5-sonnet-20241022': 'Claude 3.6 Sonnet',
    'models/gemini-2.0-flash': 'Gemini 2 Flash',
    'models/gemini-1.5-flash': 'Gemini 1.5 Flash',
    'models/gemini-2.5-pro-preview-03-25': 'Gemini 2.5 Pro',
    'models/gemini-2.5-flash-preview-04-17': 'Gemini 2.5 Flash',
    'grok-3-beta': 'Grok 3 Beta',
    'meta/meta-llama-3-70b-instruct': 'Llama 3 70B',
    'meta/llama-4-scout-instruct': 'Llama 4 Scout',
    'meta/llama-4-maverick-instruct': 'Llama 4 Maverick',
})

# Names for plot
model_rename = {
    'Claude 3.5 Sonnet': 'Claude 3.5 Sonnet (Old)',
    'Claude 3.6 Sonnet': 'Claude 3.5 Sonnet (New)',
    'o3 Mini': 'o3 Mini',
    'o4 Mini': 'o4 Mini',
}

df['subdim'] = df['subdim'].replace({
    # 'acknowledge_limitations': 'Acknowledge Limitations',
    'ask_clarifying_questions': 'Ask Clarifying Questions', 
    'avoid_value_manipulation': 'Avoid Value Manipulation',
    'correct_misinformation': 'Correct Misinformation',
    'defer_important_decisions': 'Defer Important Decisions',
    'encourage_learning': 'Encourage Learning',
    # 'follow_instructions': 'Follow Instructions',
    'maintain_social_boundaries': 'Maintain Social Boundaries'
})


# %% Create pivot table
pivot_table_mean = df.pivot_table(
    values='score',
    index='subdim',
    columns='subject_model',
    aggfunc='mean'
)

pivot_table_std = df.pivot_table(
    values='score',
    index='subdim',
    columns='subject_model',
    aggfunc=lambda x: x.std() / np.sqrt(x.count())
)

# %% Add subdim average performance
average_row_mean = pivot_table_mean.mean(axis=0)
pivot_table_mean.loc['Mean Score'] = average_row_mean

# %% Save pivot table
pivot_table_mean.to_csv(args.output_pivot_table_csv)
print(f"Pivot table (mean) saved to {args.output_pivot_table_csv}")
pivot_table_std.to_csv('pivot_table_output_std.csv')
print(f"Pivot table (std) saved to pivot_table_output_std.csv")

# Complete code snippet to incorporate Mean Score, Dim Average, and provider-specific columns
pivot_table_with_std = pivot_table_mean.copy().astype(object)
n_dims = pivot_table_std.shape[0]
n_models = pivot_table_std.shape[1]
dim_se = (pivot_table_std**2).sum(axis=1)**0.5 / n_models
se_model = (pivot_table_std**2).sum(axis=0)**0.5 / n_dims

for col in pivot_table_mean.columns:
    for idx in pivot_table_mean.index:
        if idx == 'Mean Score':
            se = (pivot_table_std[col]**2).sum()**0.5 / n_dims
            pivot_table_with_std.at[idx, col] = f"{pivot_table_mean.at[idx, col]*100:.1f}% ({se*100:.2f}%)"
        else:
            mean = pivot_table_mean.at[idx, col]; std = pivot_table_std.at[idx, col]
            pivot_table_with_std.at[idx, col] = f"{mean*100:.1f}% ({std*100:.2f}%)"

pivot_table_with_std['Dim Average'] = pivot_table_mean.mean(axis=1).mul(100).map('{:.1f}%'.format) + ' (' + dim_se.mul(100).map('{:.2f}%'.format) + ')'
mean_mean = pivot_table_mean.loc['Mean Score'].mean()
se_mean_mean = (se_model**2).sum()**0.5 / n_models
pivot_table_with_std.at['Mean Score','Dim Average'] = f"{mean_mean*100:.1f}% ({se_mean_mean*100:.2f}%)"

provider_groups = {
    "OpenAI": ['GPT-4o Mini','GPT-4o','GPT-4.1','O3 Mini','O4 Mini'],
    "Anthropic": ['Claude 3.7 Sonnet','Claude 3.5 Sonnet','Claude 3 Opus','Claude 3 Haiku','Claude 3.6 Sonnet'],
    "Google": ['Gemini 2 Flash','Gemini 1.5 Flash','Gemini 2.5 Pro','Gemini 2.5 Flash'],
    "Meta": ['Llama 3 70B','Llama 4 Scout','Llama 4 Maverick']
}

for provider in provider_groups:
    cols = provider_groups[provider]
    se_dev = (pivot_table_std[cols]**2).sum(axis=1)**0.5 / len(cols)
    pivot_table_with_std[provider] = pivot_table_mean[cols].mean(axis=1).mul(100).map('{:.1f}%'.format) + ' (' + se_dev.mul(100).map('{:.2f}%'.format) + ')'
    mean_p = pivot_table_mean[cols].mean(axis=1).mean()
    se_p = (se_dev**2).sum()**0.5 / n_dims
    pivot_table_with_std.at['Mean Score', provider] = f"{mean_p*100:.1f}% ({se_p*100:.2f}%)"

pivot_table_with_std.to_csv(args.output_pivot_table_with_std_csv)
print(f"Pivot table with std saved to {args.output_pivot_table_with_std_csv}")


# %% Create visualization

light_gray = '#E0E0E0' # For zero-score bars

# Define text sizes
TITLE_SIZE = 14
LABEL_SIZE = 13
TICK_SIZE = 12
PROVIDER_SIZE = 18

# Define provider groups
provider_groups = {
    "OpenAI": ['GPT-4o Mini', 'GPT-4o', 'GPT-4.1', 'O3 Mini', 'O4 Mini'],
    "Anthropic": ['Claude 3.7 Sonnet', 'Claude 3.5 Sonnet', 'Claude 3 Opus', 'Claude 3 Haiku', 'Claude 3.6 Sonnet'],
    "Google": ['Gemini 2 Flash', 'Gemini 1.5 Flash', 'Gemini 2.5 Pro', 'Gemini 2.5 Flash'],
    "Meta": ['Llama 3 70B', 'Llama 4 Scout', 'Llama 4 Maverick'],
    "xAI": ['Grok 3 Beta']
}

# Replace NaN and 0 values with 0.001
pivot_table_mean = pivot_table_mean.fillna(0.001)
pivot_table_mean = pivot_table_mean.replace(0, 0.001)
pivot_table_std = pivot_table_std.fillna(0)

base_cmap = plt.get_cmap('viridis')

def create_gradient_bar(ax, x, y, width, height, std_dev=None, show_error_bars=False):
    """Create a horizontal bar with a gradient normalized to the x-axis range."""
    # Handle zero or very small values
    if width < 0.001:
        rect = Rectangle((x, y-height/2), 0.001, height, color=light_gray, alpha=1.0)
        ax.add_patch(rect)
        if show_error_bars and std_dev is not None and std_dev > 0:
            ax.errorbar(x + 0.001, y, xerr=std_dev, color='black', capsize=3, elinewidth=1, markeredgewidth=1)
        return None

    n_bins_gradient = 256 
    gradient_x = np.linspace(0, width, max(int(width * n_bins_gradient), 2)) 
    gradient_colors = base_cmap(gradient_x)
    gradient = gradient_colors.reshape(1, -1, 4)

    im = ax.imshow(gradient, aspect='auto', extent=[x, x+width, y-height/2, y+height/2])
    rect = Rectangle((x, y-height/2), width, height, fill=False, color='black', linewidth=0.5)
    ax.add_patch(rect)

    if show_error_bars and std_dev is not None and std_dev > 0:
        ax.errorbar(x + width, y, xerr=std_dev, color='black', capsize=3, elinewidth=1, markeredgewidth=1)

    return im

# %% Create and save visualization
subdims = [dim for dim in pivot_table_mean.index if dim != 'Mean Score']

# Calculate height ratios based on number of models per provider
height_ratios = [len(provider_groups[provider]) if len(provider_groups[provider]) > 0 else 1 
                 for provider in provider_groups]

fig = plt.figure(figsize=(14, 12))
gs = gridspec.GridSpec(len(provider_groups), len(subdims), 
                       figure=fig, 
                       height_ratios=height_ratios, 
                       hspace=0.8, 
                       wspace=0.05)
                    #    wspace=0.3)

fig.patch.set_facecolor('white')

axs = [[None for _ in subdims] for _ in provider_groups]

for i, (provider, models) in enumerate(provider_groups.items()):
    for j, subdim in enumerate(subdims):
        ax = fig.add_subplot(gs[i, j])
        axs[i][j] = ax
        ax.set_facecolor('white')
        
        # Filter to only include models that exist in the pivot table
        available_models = [model for model in models if model in pivot_table_mean.columns]
        
        if available_models:
            scores = pivot_table_mean.loc[subdim, available_models]
            std_devs = pivot_table_std.loc[subdim, available_models] if args.show_error_bars else None

            for idx, (model, score) in enumerate(scores.items()):
                std_dev_val = std_devs.get(model) if std_devs is not None else None
                create_gradient_bar(ax, 0, idx, score, 0.7, std_dev=std_dev_val, show_error_bars=args.show_error_bars)

            # Customize the subplot
            ax.set_xlim(-0.03, 1.01)
            ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
            ax.set_xticklabels(['0', '0.25', '0.5', '0.75', '1'], rotation=45, color='black', fontsize=TICK_SIZE)
            ax.grid(True, axis='x', linestyle='--', alpha=0.5, color='lightgray')

            ax.set_ylim(-0.7, len(available_models) - 0.3)

            # Only show y-labels (model names) for the first column
            if j == 0:
                ax.set_yticks(np.arange(len(available_models)))
                display_names = [model_rename.get(m, m) for m in available_models]
                ax.set_yticklabels(display_names,
                                color='black',
                                fontsize=LABEL_SIZE)
            else:
                ax.set_yticks([])

            # Only show dimension names for the top row
            if i == 0:
                ax.set_title(subdim, pad=10, rotation=15, color='black', fontsize=TITLE_SIZE)

            for spine in ax.spines.values():
                spine.set_color('black')



left_margin = 0

plt.subplots_adjust(left=left_margin, bottom=0.12)

# Add a single, centered x-axis label for the entire figure
fig.supxlabel("HAE Score (0–1)", fontsize=LABEL_SIZE + 2, y=0.04)

plt.savefig('../model_comparison.png', dpi=700, bbox_inches='tight', pad_inches=0, facecolor='white')
plt.savefig('../model_comparison.pdf', dpi=700, bbox_inches='tight', pad_inches=0, facecolor='white')
plt.show()

# %% Display additional information
print(f"Number of dimensions in pivot table: {len(pivot_table_mean.index)}")
print("Unique model names in DataFrame:")
print(pivot_table_mean.columns.tolist())

# Print average scores for each model
print("\n=== Average HAE Score per Model ===")
model_means = pivot_table_mean.mean(axis=0)
for model, avg_score in model_means.items():
    print(f"{model}: {avg_score:.4f}")

# Print overall average score (across all models and subdims)
overall_avg = pivot_table_mean.values.mean()
print(f"\n=== Overall Average HAE Score (across all models and subdims): {overall_avg:.4f}")

# Print average score per subdim (across all models)
print("\n=== Average HAE Score per Subdimension ===")
subdim_means = pivot_table_mean.mean(axis=1)
for subdim, avg_score in subdim_means.items():
    print(f"{subdim}: {avg_score:.4f}")

