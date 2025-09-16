"""High-level plotting functions that orchestrate different plot types."""

import os
import pandas as pd
import matplotlib.pyplot as plt
from itertools import product
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from matplotlib.colorbar import make_axes

from .data_processing import gather_fis, format_rois, get_fullrang_minmax, absmax
from .brain_plots import (
    broadcast_to_fsaverage,
    draw_plot,
    get_global_minmax,
    make_roi_model_plot,
)
from .statistical_plots import make_paper_effectsize_plot


def make_collage_plot(
    fis_agg: dict,
    targets: list,
    correct: list,
    conditions: list,
    params: dict,
    title: str,
    roi=False,
):
    """Make collage plot of feature importance maps."""

    n_targets = len(targets)
    n_conditions = len(conditions)
    n_correct = len(correct)

    if roi:
        fig, axs = plt.subplots(
            n_targets,
            n_conditions * n_correct,
            figsize=(4 * n_conditions * n_correct, 3 * n_targets),
        )

        plot_df = format_rois(fis_agg, correct, conditions)
        vmin, vmax = get_fullrang_minmax(plot_df["fis"])

        for i, target in enumerate(targets):
            for j, (cor, cond) in enumerate(product(correct, conditions)):
                ax_idx = j
                if n_targets == 1:
                    ax = axs[ax_idx] if n_conditions * n_correct > 1 else axs
                else:
                    ax = axs[i, ax_idx] if n_conditions * n_correct > 1 else axs[i]

                make_roi_model_plot(plot_df, target, ax, vmin, vmax)

                if i == 0:
                    ax.set_title(f"{cor}_{cond}", fontsize=12, fontweight="bold")
                if j == 0:
                    ax.set_ylabel(target, fontsize=12, fontweight="bold")
    else:
        fig, axs = plt.subplots(
            n_targets,
            n_conditions * n_correct * 2,
            figsize=(6 * n_conditions * n_correct, 3 * n_targets),
        )

        for i, target in enumerate(targets):
            for j, (cor, cond) in enumerate(product(correct, conditions)):
                lab = f"{target}_{cor}_{cond}"

                if lab not in fis_agg.keys():
                    continue

                fis_avg = fis_agg[lab]
                plot_df = broadcast_to_fsaverage(fis_avg)

                lh = plot_df[plot_df["hemi"] == "lh"]["data"]
                rh = plot_df[plot_df["hemi"] == "rh"]["data"]

                vmin, vmax = get_global_minmax(lh, rh)

                ax_start = j * 2
                if n_targets == 1:
                    ax_pair = [axs[ax_start], axs[ax_start + 1]]
                else:
                    ax_pair = [axs[i, ax_start], axs[i, ax_start + 1]]

                draw_plot(
                    lh, rh, ax_pair, "avg", vmin=vmin, vmax=vmax, avg_method=absmax
                )

                if i == 0:
                    ax_pair[0].set_title(f"{cor}_{cond}_LH", fontsize=10)
                    ax_pair[1].set_title(f"{cor}_{cond}_RH", fontsize=10)

    plt.suptitle(title, fontsize=16, fontweight="bold")
    plt.tight_layout()

    output_dir = params.get("output_dir", "./plots")
    os.makedirs(output_dir, exist_ok=True)
    filename = title.lower().replace(" ", "_").replace(":", "")
    plt.savefig(f"{output_dir}/{filename}.png", dpi=300, bbox_inches="tight")
    plt.show()


def make_colorbar(
    vmin: float, vmax: float, cmap: str, ax, orientation="horizontal", shrink=0.8
):
    """Make standalone colorbar."""

    norm = Normalize(vmin=vmin, vmax=vmax)
    sm = ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])

    if orientation == "horizontal":
        cbar_ax, _ = make_axes(ax, location="bottom", size="5%", pad=0.1)
    else:
        cbar_ax, _ = make_axes(ax, location="right", size="5%", pad=0.1)

    cbar = plt.colorbar(sm, cax=cbar_ax, orientation=orientation, shrink=shrink)
    return cbar


def produce_effectsize_plot(params: dict, model: str):
    """Produce effect size plot from model results."""

    results_path = params["results_path"]
    results_files = [f for f in os.listdir(results_path) if f.endswith(f"_{model}.pkl")]

    all_results = []
    for file in results_files:
        result = pd.read_pickle(os.path.join(results_path, file))
        all_results.append(result)

    combined_df = pd.concat(all_results, ignore_index=True)
    make_paper_effectsize_plot(combined_df, params)


def produce_supplement_plots(params: dict):
    """Produce supplementary plots."""

    models = ["ridge", "lasso", "elastic"]

    for model in models:
        produce_effectsize_plot(params, model)

        fis_path = params.get("fis_path")
        if fis_path and os.path.exists(fis_path):
            fis_data = pd.read_pickle(fis_path)
            gather_fis([fis_data], compare_scopes=False)

            targets = ["beta", "gamma", "tau", "v", "a", "t0"]
            correct = ["correct", "notf"]
            conditions = [
                "all",
                "correct_go",
                "correct_stop",
                "incorrect_go",
                "incorrect_stop",
            ]

            make_collage_plot(
                fis_data,
                targets,
                correct,
                conditions,
                params,
                f"Feature Importance: {model.title()} Model",
            )


def produce_plots(params: dict, model: str, compare_scopes=False):
    """Produce all plots for a given model."""

    produce_effectsize_plot(params, model)

    if params.get("produce_fis_plots", True):
        fis_path = params.get("fis_path")
        if fis_path and os.path.exists(fis_path):
            fis_data = pd.read_pickle(fis_path)
            gather_fis([fis_data], compare_scopes=compare_scopes)

            targets = ["beta", "gamma", "tau", "v", "a", "t0"]
            correct = ["correct", "notf"]
            conditions = [
                "all",
                "correct_go",
                "correct_stop",
                "incorrect_go",
                "incorrect_stop",
            ]

            make_collage_plot(
                fis_data,
                targets,
                correct,
                conditions,
                params,
                f"Feature Importance: {model.title()} Model",
                roi=params.get("roi_mode", False),
            )
