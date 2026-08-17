#!/usr/bin/env python3
"""Plot HAB scores for every evaluated model, grouped by developer.

Reproduces the layout of Figure 4 in the paper (horizontal gradient bars per
dimension, rows grouped by developer, error bars = standard error) for the
full 38-model leaderboard: the 25 models in the paper plus 13 models added in
May 2026, all judged by o3-2025-04-16 on the same fixed 3,000-prompt set.

Input:  results_summary_38_models.csv  (subject_model, dimension, mean, se, n)
        mean/se are on the 0-1 scale (rubric score / 10), n = prompts per cell.
Output: hab_all_models.png / .pdf

Usage:  python figures/plot_all_models.py
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Rectangle

HERE = os.path.dirname(os.path.abspath(__file__))
INPUT_CSV = os.path.join(HERE, "results_summary_38_models.csv")
OUT_STEM = os.path.join(HERE, "hab_all_models")

DIMENSIONS = [
    ("ask_clarifying_questions", "Ask Clarifying Questions"),
    ("avoid_value_manipulation", "Avoid Value Manipulation"),
    ("correct_misinformation", "Correct Misinformation"),
    ("defer_important_decisions", "Defer Important Decisions"),
    ("encourage_learning", "Encourage Learning"),
    ("maintain_social_boundaries", "Maintain Social Boundaries"),
]

# Models added after the paper (May 2026 rerun) are marked with an asterisk.
NEW_MODELS = {
    "claude-opus-4-7", "claude-sonnet-4-6", "claude-haiku-4-5-20251001",
    "claude-opus-4-6", "claude-opus-4-5-20251101",
    "gpt-5.5", "gpt-5.4", "gpt-5.4-mini",
    "google/gemini-3.1-pro-preview", "deepseek/deepseek-v3.2", "z-ai/glm-5.1",
    "moonshotai/kimi-k2.6", "deepseek-v4-pro",
}

# subject_model id -> (display name, developer)
MODELS = {
    # Anthropic
    "claude-3-haiku-20240307": ("Claude 3 Haiku", "Anthropic"),
    "claude-3-opus-20240229": ("Claude 3 Opus", "Anthropic"),
    "claude-3-5-haiku-20241022": ("Claude 3.5 Haiku", "Anthropic"),
    "claude-3-5-sonnet-20240620": ("Claude 3.5 Sonnet (Old)", "Anthropic"),
    "claude-3-5-sonnet-20241022": ("Claude 3.5 Sonnet (New)", "Anthropic"),
    "claude-3-7-sonnet-20250219": ("Claude 3.7 Sonnet", "Anthropic"),
    "claude-sonnet-4-20250514": ("Claude Sonnet 4", "Anthropic"),
    "claude-opus-4.1-20250805": ("Claude Opus 4.1", "Anthropic"),
    "claude-haiku-4-5-20251001": ("Claude Haiku 4.5", "Anthropic"),
    "claude-opus-4-5-20251101": ("Claude Opus 4.5", "Anthropic"),
    "claude-sonnet-4-6": ("Claude Sonnet 4.6", "Anthropic"),
    "claude-opus-4-6": ("Claude Opus 4.6", "Anthropic"),
    "claude-opus-4-7": ("Claude Opus 4.7", "Anthropic"),
    # OpenAI
    "gpt-4o": ("GPT-4o", "OpenAI"),
    "gpt-4.1": ("GPT-4.1", "OpenAI"),
    "gpt-4.1-mini": ("GPT-4.1 Mini", "OpenAI"),
    "o3-mini-2025-01-31": ("o3 Mini", "OpenAI"),
    "o3-2025-04-16": ("o3", "OpenAI"),
    "o4-mini-2025-04-16": ("o4 Mini", "OpenAI"),
    "gpt-5": ("GPT-5", "OpenAI"),
    "gpt-5-high": ("GPT-5 (high)", "OpenAI"),
    "gpt-5.4": ("GPT-5.4", "OpenAI"),
    "gpt-5.4-mini": ("GPT-5.4 Mini", "OpenAI"),
    "gpt-5.5": ("GPT-5.5", "OpenAI"),
    # Google
    "gemini-1.5-flash": ("Gemini 1.5 Flash", "Google"),
    "gemini-2.0-flash": ("Gemini 2 Flash", "Google"),
    "gemini-2.5-flash-preview-04-17": ("Gemini 2.5 Flash", "Google"),
    "gemini-2.5-pro-preview-03-25": ("Gemini 2.5 Pro", "Google"),
    "google/gemini-3.1-pro-preview": ("Gemini 3.1 Pro", "Google"),
    # Meta
    "meta-llama-3-70b-instruct": ("Llama 3 70B", "Meta"),
    "llama-4-scout-instruct": ("Llama 4 Scout", "Meta"),
    "llama-4-maverick-instruct": ("Llama 4 Maverick", "Meta"),
    # xAI
    "grok-3": ("Grok 3", "xAI"),
    "x-ai/grok-4": ("Grok 4", "xAI"),
    # DeepSeek
    "deepseek/deepseek-v3.2": ("DeepSeek V3.2", "DeepSeek"),
    "deepseek-v4-pro": ("DeepSeek V4 Pro", "DeepSeek"),
    # Moonshot AI
    "moonshotai/kimi-k2.6": ("Kimi K2.6", "Moonshot AI"),
    # Zhipu AI
    "z-ai/glm-5.1": ("GLM-5.1", "Zhipu AI"),
}
DEVELOPER_ORDER = ["Anthropic", "OpenAI", "Google", "DeepSeek", "Moonshot AI", "Zhipu AI", "Meta", "xAI"]

TITLE_SIZE, LABEL_SIZE, TICK_SIZE, PROVIDER_SIZE = 13, 11, 10, 15
LIGHT_GRAY = "#E0E0E0"
CMAP = plt.get_cmap("viridis")


def gradient_bar(ax, y, width, se, height=0.7):
    """Horizontal bar filled with a viridis gradient normalised to the 0-1 axis."""
    if width < 0.001:
        ax.add_patch(Rectangle((0, y - height / 2), 0.001, height, color=LIGHT_GRAY))
        return
    xs = np.linspace(0, width, max(int(width * 256), 2))
    ax.imshow(CMAP(xs).reshape(1, -1, 4), aspect="auto",
              extent=[0, width, y - height / 2, y + height / 2])
    ax.add_patch(Rectangle((0, y - height / 2), width, height, fill=False, color="black", linewidth=0.5))
    if se > 0:
        ax.errorbar(width, y, xerr=se, color="black", capsize=2, elinewidth=0.8, markeredgewidth=0.8)


def main():
    df = pd.read_csv(INPUT_CSV)
    unknown = set(df.subject_model) - set(MODELS)
    assert not unknown, f"models missing from MODELS map: {unknown}"
    assert set(df.dimension) == {d for d, _ in DIMENSIONS}
    assert (df.n == 500).all(), "expected 500 prompts per model x dimension"

    mean = df.pivot(index="subject_model", columns="dimension", values="mean")
    se = df.pivot(index="subject_model", columns="dimension", values="se")
    mean["overall"] = mean[[d for d, _ in DIMENSIONS]].mean(axis=1)
    se["overall"] = np.sqrt((se[[d for d, _ in DIMENSIONS]] ** 2).sum(axis=1)) / len(DIMENSIONS)
    columns = [("overall", "Overall (mean of 6)")] + DIMENSIONS

    # Rows: developers in fixed order; models within a developer sorted by overall score.
    groups = {}
    for dev in DEVELOPER_ORDER:
        ids = [m for m, (_, d) in MODELS.items() if d == dev and m in mean.index]
        groups[dev] = sorted(ids, key=lambda m: -mean.loc[m, "overall"])
    assert sum(len(v) for v in groups.values()) == len(mean) == 38

    n_rows = sum(len(v) for v in groups.values())
    fig = plt.figure(figsize=(17, 0.34 * n_rows + 2.2))
    gs = gridspec.GridSpec(len(groups), len(columns), figure=fig,
                           height_ratios=[len(v) for v in groups.values()],
                           hspace=0.55, wspace=0.06)
    fig.patch.set_facecolor("white")

    for i, (dev, ids) in enumerate(groups.items()):
        for j, (col, col_title) in enumerate(columns):
            ax = fig.add_subplot(gs[i, j])
            for k, m in enumerate(ids):
                gradient_bar(ax, k, mean.loc[m, col], se.loc[m, col])
            ax.set_xlim(-0.03, 1.01)
            ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
            ax.set_xticklabels(["0", "0.25", "0.5", "0.75", "1"], rotation=45, fontsize=TICK_SIZE)
            ax.grid(True, axis="x", linestyle="--", alpha=0.5, color="lightgray")
            ax.set_ylim(-0.7, len(ids) - 0.3)
            ax.invert_yaxis()  # best model at the top of each group
            if j == 0:
                ax.set_yticks(np.arange(len(ids)))
                ax.set_yticklabels([MODELS[m][0] + ("*" if m in NEW_MODELS else "") for m in ids],
                                   fontsize=LABEL_SIZE)
                ax.set_ylabel(dev, fontsize=PROVIDER_SIZE, fontweight="bold", rotation=0,
                              ha="right", va="center", labelpad=110)
            else:
                ax.set_yticks([])
            if i == 0:
                ax.set_title(col_title, pad=10, rotation=15, fontsize=TITLE_SIZE)
            if col == "overall":
                ax.set_facecolor("#F5F5F5")

    fig.subplots_adjust(bottom=0.06, top=0.94)
    fig.supxlabel("HAB Score (0–1). Error bars: standard error over 500 prompts per cell. "
                  "* = added May 2026, not in the paper.", fontsize=LABEL_SIZE + 1, y=0.02)
    fig.savefig(OUT_STEM + ".png", dpi=200, bbox_inches="tight", facecolor="white")
    fig.savefig(OUT_STEM + ".pdf", bbox_inches="tight", facecolor="white")
    print("wrote", OUT_STEM + ".png")

    # Sanity: overall column matches the leaderboard CSV convention (mean of dims x 100).
    print(mean["overall"].sort_values(ascending=False).mul(100).round(1).to_string())


if __name__ == "__main__":
    main()
