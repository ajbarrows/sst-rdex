"""Statistical plotting utilities."""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr


def make_effect_compare_plot(
    plot_df: pd.DataFrame,
    axs,
    vmin=None,
    vmax=None,
    cmap="bwr",
    show_cbar=True,
):
    """Make effect comparison plot."""

    target_order = ["beta", "gamma", "tau", "v", "a", "t0"]
    condition_order = [
        "all",
        "correct_go",
        "correct_stop",
        "incorrect_go",
        "incorrect_stop",
    ]
    correct_order = ["correct", "notf"]

    col_labels = []
    for correct in correct_order:
        for condition in condition_order:
            col_labels.append(f"{correct}_{condition}")

    plot_df_pivot = plot_df.pivot_table(
        index="target", columns=["correct", "condition"], values="mean_scores_r2"
    )

    plot_df_pivot = plot_df_pivot.reindex(index=target_order)
    plot_df_pivot.columns = [f"{col[0]}_{col[1]}" for col in plot_df_pivot.columns]
    plot_df_pivot = plot_df_pivot.reindex(columns=col_labels)

    sns.heatmap(
        plot_df_pivot,
        ax=axs,
        cmap=cmap,
        center=0,
        square=False,
        vmin=vmin,
        vmax=vmax,
        cbar=show_cbar,
        xticklabels=True,
        yticklabels=True,
        annot=True,
        fmt=".3f",
    )

    axs.set_xlabel("")
    axs.set_ylabel("")


def make_paper_effectsize_plot(df, params):
    """Make paper effect size plot."""

    f, axs = plt.subplots(figsize=(16, 6))

    plot_df = df[df["scope"] == "mri_confounds"].copy()

    vmin, vmax = plot_df["mean_scores_r2"].min(), plot_df["mean_scores_r2"].max()

    make_effect_compare_plot(
        plot_df,
        axs,
        vmin=vmin,
        vmax=vmax,
        cmap="viridis",
        show_cbar=True,
    )

    axs.set_title("R² Scores by Target and Condition", fontsize=16, fontweight="bold")

    plt.tight_layout()

    output_dir = params.get("output_dir", "./plots")
    plt.savefig(
        f"{output_dir}/effect_size_comparison.png", dpi=300, bbox_inches="tight"
    )
    plt.show()


def make_corplot(x, y, fontsize, xlab=None, ylab=None, color="purple", ax=None):
    """Make correlation plot."""

    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))

    ax.scatter(x, y, color=color, alpha=0.6, s=20)

    r, p = pearsonr(x, y)

    ax.text(
        0.05,
        0.95,
        f"r = {r:.3f}\np = {p:.3f}",
        transform=ax.transAxes,
        fontsize=fontsize,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )

    z = np.polyfit(x, y, 1)
    p_fit = np.poly1d(z)
    ax.plot(x, p_fit(x), color="black", linestyle="--", alpha=0.8)

    if xlab:
        ax.set_xlabel(xlab, fontsize=fontsize)
    if ylab:
        ax.set_ylabel(ylab, fontsize=fontsize)

    ax.tick_params(axis="both", which="major", labelsize=fontsize - 2)

    return ax


def make_feat_imp_radar_plot(df, ax, legend=True):
    """Make feature importance radar plot."""

    categories = df["target"].unique()
    N = len(categories)

    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]

    for scope in df["scope"].unique():
        values = df[df["scope"] == scope]["mean_scores_r2"].tolist()
        values += values[:1]

        ax.plot(angles, values, "o-", linewidth=2, label=scope)
        ax.fill(angles, values, alpha=0.25)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories)
    ax.set_ylim(0, df["mean_scores_r2"].max() * 1.1)

    if legend:
        ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.0))

    return ax
