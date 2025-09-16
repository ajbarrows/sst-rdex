import os

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import neurotools.plotting as ntp
import seaborn as sns

from joblib import Parallel, delayed
from itertools import repeat

from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from matplotlib.colorbar import make_axes

from nilearn.datasets import fetch_atlas_surf_destrieux

from neurotools.plotting.ref import SurfRef

from abcd_tools.image.preprocess import map_hemisphere

from scipy.stats import false_discovery_control, pearsonr
from itertools import product


def relabel_plotting_data(df, process_map, target_map, color_map):
    """Relabel data for plotting.

    Args:
        df (pd.DataFrame): Dataframe.
        process_map (dict): Process map.
        target_map (dict): Target map.
        color_map (dict): Color map.

    Returns:
        pd.DataFrame: Relabeled dataframe
    """

    df["scope"] = df["scope"].fillna("all")
    df = df[df["scope"] != "cov + mri_confounds"]
    df["scope"] = df["scope"].str.replace("cov \+ ", "", regex=True)
    df.loc[:, "process"] = df["target"]
    df["process"] = df["process"].replace(process_map)

    df.loc[:, "color"] = df["process"]
    df["color"] = df["color"].replace(color_map)
    df["target"] = df["target"].replace(target_map)

    return df


def sort(df: pd.DataFrame) -> pd.DataFrame:
    """Sort dataframe.

    Args:
        df (pd.DataFrame): Dataframe.

    Returns:
        pd.DataFrame: Sorted dataframe.
    """
    avg = (
        df[["target", "mean_scores_r2", "std_scores_r2"]]
        .groupby("target")
        .mean(numeric_only=True)
        .sort_values("mean_scores_r2", ascending=False)
    )
    avg.columns = ["avg_mean", "avg_std"]
    df = (
        df.set_index("target")
        #   .drop(columns=['test_r2'])
        .join(avg)
        .sort_values(
            by=["process", "avg_mean", "mean_scores_r2"], ascending=[True, False, False]
        )
        .reset_index()
        .drop(columns=["avg_mean", "avg_std"])
    )
    return df


def make_effect_compare_plot(
    df: pd.DataFrame, model: str, title: str, fpath: str
) -> None:
    """Make effect compare plot.
    Args:

        df (pd.DataFrame): Dataframe.
        model (str): Model ['ridge', 'elastic'].
        title (str): Title.
        fpath (str): File path.
    """
    sns.set_context("paper", font_scale=1.25)

    hatches = ["", "/", "-", "X", "O"]

    scope_map = {"ridge": "Ridge Regression", "lasso": "Lasso Regression"}

    fig, ax = plt.subplots(figsize=(15, 5))

    # greypallete = np.repeat('lightgrey', len(df))
    if "scope" not in df.columns:
        df["scope"] = "all"

    n_scopes = len(df["scope"].drop_duplicates())
    greypallete = list(np.repeat("lightgrey", n_scopes))

    order = df["target"].drop_duplicates()
    df = df.replace(scope_map)

    g = sns.barplot(
        x="target",
        y="mean_scores_r2",
        hue="scope",
        data=df,
        palette=greypallete,
        # palette=palette,
        order=order,
    )

    g.legend_.set_title("")

    ax.grid(linestyle=":")
    bars = ax.patches[: len(ax.patches) - n_scopes]
    x_coords = [p.get_x() + 0.5 * p.get_width() for p in bars]
    y_coords = [p.get_height() for p in bars]

    ax.errorbar(x=x_coords, y=y_coords, yerr=df["std_scores_r2"], fmt="none", c="k")

    # only want one set of colors
    palette = df[["target", "color"]].drop_duplicates()["color"]

    for bars, hatch, legend_handle in zip(
        ax.containers, hatches, ax.legend_.legendHandles
    ):
        for bar, color in zip(bars, palette):
            bar.set_facecolor(color)
            bar.set_hatch(hatch)
        legend_handle.set_hatch(hatch + hatch)

    sns.pointplot(
        x="target",
        y="test_r2",
        data=df,
        hue="scope",
        markersize=2,
        dodge=0.5 if n_scopes > 1 else 0,
        linestyles="none",
        palette=greypallete,
        order=order,
        legend=False,
    )

    # formatting
    ax.set(xlabel=None)
    ax.set(ylabel="Avg. $R^{2}$")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
    ax.spines[["top", "right"]].set_visible(False)

    fig.subplots_adjust(top=0.9)
    fig.suptitle(title)

    fpath = f"{fpath}.png"
    plt.savefig(fpath, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Plot saved to {fpath}")


def make_paper_effectsize_plot(df, params):

    process_map = params["process_map"]
    target_map = params["target_map"]
    params["color_map"]

    go_targets = [target_map[k] for k, v in process_map.items() if v == "Go Process"]
    stop_targets = [
        target_map[k] for k, v in process_map.items() if v == "Stop Process"
    ]
    ssrt_targets = [target_map[k] for k, v in process_map.items() if v == "SSRT"]

    empirical_keys = ["correct_go_mrt", "correct_go_stdrt", "issrt"]
    empirical = [target_map[k] for k in empirical_keys]

    sns.set_context("paper", font_scale=1.75)
    fig, ax = plt.subplots(
        1,
        4,
        figsize=(20, 8),
        sharey=True,
        gridspec_kw={"width_ratios": [7, 2.5, 1, 0.05]},
    )

    test_marker_size = 8

    fig.suptitle("Cross-Validated Ridge Regression Model Performance")
    # Go Process

    # go_color = '#06A94D'
    go_color = "#77DD77"

    x_loc = 0
    starting_points = []

    go_targets = df[df["target"].isin(go_targets)]["target"].values

    for target in go_targets:

        if target in empirical:
            hatch = "/"
        else:
            hatch = None

        starting_points.append(x_loc)

        row = df[(df["target"] == target)]

        ax[0].bar(
            x_loc,
            height=row["mean_scores_r2"],
            yerr=row["std_scores_r2"],
            width=1 / 2,
            color=go_color,
            hatch=hatch,
        )
        ax[0].plot(
            x_loc,
            row["test_r2"],
            marker="o",
            markersize=test_marker_size,
            color="lightgrey",
        )

        x_loc += 1

    ax[0].grid(linestyle=":")
    ax[0].set_xticks(np.array(starting_points))
    ax[0].set_xticklabels(go_targets, rotation=45, ha="right")
    ax[0].set_title("Go Process")
    ax[0].set_ylabel("Model $R^2$")

    ax[0].spines[["top", "right"]].set_visible(False)

    # Stop process
    # stop_color = '#DF2C14'
    stop_color = "lightcoral"
    stop_targets = df[df["target"].isin(stop_targets)]["target"].values

    x_loc = 0
    starting_points = []
    for target in stop_targets:

        starting_points.append(x_loc)

        row = df[(df["target"] == target)]

        ax[1].bar(
            x_loc,
            height=row["mean_scores_r2"],
            yerr=row["std_scores_r2"],
            width=0.5,
            color=stop_color,
        )
        ax[1].plot(
            x_loc,
            row["test_r2"],
            marker="o",
            markersize=test_marker_size,
            color="lightgrey",
        )

        x_loc += 1

    ax[1].grid(linestyle=":")
    ax[1].set_xticks(np.array(starting_points))

    ax[1].set_xticklabels(stop_targets, ha="right", rotation=45)
    ax[1].set_title("Stop Process")

    ax[1].spines[["top", "left", "right"]].set_visible(False)

    # SSRT
    ssrt_color = "#797EF6"
    ssrt_targets = df[df["target"].isin(ssrt_targets)]["target"].values

    x_loc = 0
    starting_points = []
    for target in ssrt_targets:

        if target in empirical:
            hatch = "/"
        else:
            hatch = None

        starting_points.append(x_loc)

        row = df[(df["target"] == target)]
        ax[2].bar(
            x_loc,
            height=row["mean_scores_r2"],
            yerr=row["std_scores_r2"],
            width=0.5,
            color=ssrt_color,
            hatch=hatch,
        )
        ax[2].plot(
            x_loc,
            row["test_r2"],
            color="lightgrey",
            marker="o",
            markersize=test_marker_size,
            alpha=0.75,
        )

        x_loc += 1

    ax[2].grid(linestyle=":")
    ax[2].set_xticks(np.array(starting_points))
    ax[2].set_xticklabels(ssrt_targets, rotation=45, ha="right")
    ax[2].set_title("SSRT")

    ax[2].spines[["top", "left"]].set_visible(False)

    # add blank space
    ax[3].axis("off")

    plt.savefig(
        params["plot_output_path"] + "ridge_model_performance.png",
        dpi=300,
        bbox_inches="tight",
    )
    print("Plot saved to " + params["plot_output_path"] + "ridge_model_performance.png")


def broadcast_to_fsaverage(fis_agg: pd.Series, n_vertices=10242) -> pd.DataFrame:
    """Broadcast feature importance to fsaverage5.

    Args:
        fis_agg (pd.Series): Feature importance.
        n_vertices (int, optional): Number of vertices. Defaults to 10242+1.

    Returns:
        pd.DataFrame: Broadcasted feature importance.
    """

    def _split_hemisphere(df):
        df = df.reset_index(names=["correct", "condition", "hemisphere"])
        lh = df[df["hemisphere"] == "lh"].drop(columns="hemisphere")
        rh = df[df["hemisphere"] == "rh"].drop(columns="hemisphere")

        return lh, rh

    fis = fis_agg.copy()

    fis.index = fis.index.str.split("_", expand=True)
    fis = fis.unstack(level=2)
    # fis = fis.unstack()

    # convert columns to integers and sort
    fis.columns = fis.columns.astype(int)
    fis = fis.reindex(sorted(fis.columns), axis=1)

    # need to insert blank columns for missing vertices
    vertex_names = [*range(1, n_vertices + 1)]
    # vertex_names = [*range(0, n_vertices)]
    null_df = pd.DataFrame(np.nan, columns=vertex_names, index=fis.index)
    null_df = null_df.drop(columns=fis.columns)

    df = fis.join(null_df, how="outer")
    lh, rh = _split_hemisphere(df)

    return lh, rh


def load_destrieux_atlas():
    """Load Destrieux atlas."""
    atlas = fetch_atlas_surf_destrieux()
    return atlas


def map_destrieux(
    lh: pd.DataFrame,
    rh: pd.DataFrame,
    prefix: str = "",
    mask_non_significant=False,
    use_fdr=False,
    alpha=0.01,
) -> pd.DataFrame:
    """Map Destrieux atlas.

    Args:
        lh (pd.DataFrame): Left hemisphere.
        rh (pd.DataFrame): Right hemisphere.
        prefix (str, optional): Prefix. Defaults to ''.

    Returns:
        pd.DataFrame: Mapped dataframe.
    """

    dest = load_destrieux_atlas()

    correct_values = pd.unique(lh["correct"])
    condition_values = pd.unique(lh["condition"])

    idx = ["correct", "condition"]

    lh_df = pd.DataFrame()
    rh_df = pd.DataFrame()

    lh_tvalues = pd.DataFrame()
    rh_tvalues = pd.DataFrame()

    lh_pvalues = pd.DataFrame()
    rh_pvalues = pd.DataFrame()

    def _assemble_df(lh_mapped, rh_mapped, lh_correct, rh_correct, lh, rh):
        lh_mapped.index = lh_correct.index
        rh_mapped.index = rh_correct.index

        lh_tmp = pd.concat([lh, lh_mapped])
        rh_tmp = pd.concat([rh, rh_mapped])

        return lh_tmp, rh_tmp

    def apply_fdr(pvalues):
        return pd.DataFrame(
            false_discovery_control(pvalues, method="by"),
            index=pvalues.index,
            columns=pvalues.columns,
        )

    for correct in correct_values:

        for condition in condition_values:

            lh_correct = lh[(lh["correct"] == correct) & (lh["condition"] == condition)]
            rh_correct = rh[(rh["correct"] == correct) & (rh["condition"] == condition)]

            lh_correct = lh_correct.set_index(idx)
            rh_correct = rh_correct.set_index(idx)

            lh_mapped, lh_t, lh_p = map_hemisphere(
                lh_correct,
                mapping=dest["map_left"],
                labels=dest["labels"],
                prefix=prefix,
                suffix=".lh",
                return_statistics=True,
            )
            rh_mapped, rh_t, rh_p = map_hemisphere(
                rh_correct,
                mapping=dest["map_right"],
                labels=dest["labels"],
                prefix=prefix,
                suffix=".rh",
                return_statistics=True,
            )

            lh_df, rh_df = _assemble_df(
                lh_mapped, rh_mapped, lh_correct, rh_correct, lh_df, rh_df
            )
            lh_tvalues, rh_tvalues = _assemble_df(
                lh_t, rh_t, lh_correct, rh_correct, lh_tvalues, rh_tvalues
            )
            lh_pvalues, rh_pvalues = _assemble_df(
                lh_p, rh_p, lh_correct, rh_correct, lh_pvalues, rh_pvalues
            )

    if use_fdr:
        lh_pvalues = apply_fdr(lh_pvalues)
        rh_pvalues = apply_fdr(rh_pvalues)

    if mask_non_significant:
        lh_tvalues = lh_tvalues.mask(lh_pvalues > alpha)
        rh_tvalues = rh_tvalues.mask(rh_pvalues > alpha)

    df = pd.concat([lh_df, rh_df], axis=1)
    vmin, vmax = get_fullrang_minmax(df)

    return lh_df.reset_index(), rh_df.reset_index(), vmin, vmax
    # return lh_df, rh_df, lh_tvalues, rh_tvalues, lh_pvalues, rh_pvalues, vmin, vmax
    # return lh_tvalues.reset_index(), rh_tvalues.reset_index(), vmin, vmax


def absmax(x):
    """Return absolute maximum value."""
    idx = np.argmax(np.abs(x))
    return x[idx]


def get_fullrang_minmax(series: pd.Series):
    """Get full range minmax.

    Args:
        series (pd.Series): Series.

    Returns:
        tuple: Minmax.
    """
    mi = series.min().min()
    ma = series.max().max()

    abs_max = max(np.abs(mi), np.abs(ma))

    return -abs_max, abs_max


def draw_plot(lh, rh, ax, mode, cmap="bwr", vmin=None, vmax=None, avg_method=absmax):
    """Draw plot.

    Args:
        lh (pd.DataFrame): Left hemisphere.
        rh (pd.DataFrame): Right hemisphere.
        ax (matplotlib.axes._subplots.AxesSubplot): Axes.
        mode (str): Mode.
        cmap (str, optional): Colormap. Defaults to 'bwr'.
        vmin (float, optional): Min value. Defaults to None.
        vmax (float, optional): Max value. Defaults to None.
        avg_method (function, optional): Average method. Defaults to absmax.

    Returns:
        None
    """

    if mode == "vertex_parcellated":
        plot_df = pd.concat([lh, rh], axis=1)
        fetch_atlas_surf_destrieux()
        surf_ref = SurfRef(space="fsaverage5", parc="destr")
        to_plot = surf_ref.get_hemis_plot_vals(plot_df)

        ntp.plot(
            to_plot, threshold=0, ax=ax, cmap=cmap, vmin=vmin, vmax=vmax, colorbar=False
        )

    else:
        plot_dict = {"lh": lh.values, "rh": rh.values}
        ntp.plot(
            plot_dict,
            avg_method=avg_method,
            ax=ax,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            colorbar=False,
            threshold=0,
        )


def absmax(x):
    idx = np.argmax(np.abs(x))
    return x[idx]


def format_rois(fis_agg: dict, correct, condition):

    tmp = fis_agg.copy()

    tmp.index = tmp.index.str.replace("tfmri_sst_all_", "")
    tmp.index = tmp.index.str.split("_beta_", expand=True)

    tmp = tmp.unstack(level=1)
    tmp.index = tmp.index.str.split("_", expand=True)
    tmp.reset_index(names=["correct", "condition"], inplace=True)

    tmp = tmp[(tmp["correct"] == correct) & (tmp["condition"] == condition)]

    remove_strings = [
        "_correct_go",
        "_correct_stop",
        "_incorrect_go",
        "_incorrect_stop",
    ]

    for name in remove_strings:
        tmp.columns = tmp.columns.str.replace(name, "")

    return tmp


def make_roi_model_plot(plot_df, target, ax, vmin, vmax, cmap="bwr"):

    fetch_atlas_surf_destrieux()
    surf_ref = SurfRef(space="fsaverage5", parc="destr")

    to_plot = surf_ref.get_hemis_plot_vals(plot_df)

    ntp.plot(
        to_plot, threshold=0, ax=ax, cmap=cmap, vmin=vmin, vmax=vmax, colorbar=False
    )


def format_for_plotting(
    fis: pd.DataFrame, correct: str, condition: str
) -> pd.DataFrame:
    tmp = fis[(fis["correct"] == correct) & (fis["condition"] == condition)]

    tmp = tmp.drop(columns=["correct", "condition"])
    tmp[np.isnan(tmp)] = 0
    return tmp


def make_collage_plot(
    fis_agg: dict,
    target,
    target_map,
    basepath="../../data/06_reporting/rdex_prediction/fis_plots",
    agg="avg_fis",
    mode="vertex",
    model="enet",
    fontsize=25,
):

    if "roi" not in model:
        lh, rh = broadcast_to_fsaverage(fis_agg[target])
        conditions = pd.unique(lh["condition"])
        n_cond = conditions.shape[0]
    else:
        conditions = ["go", "stop"]
        n_cond = len(conditions)

    directions = ["correct", "", "incorrect"]
    width_ratios = [1]
    height_ratios = [100, 1, 100]

    col_ratios = list(repeat(10, n_cond))

    width_ratios.extend(col_ratios)
    width_ratios.extend([2])  # colorbar

    gs = {
        "width_ratios": width_ratios,
        "height_ratios": height_ratios,
        "hspace": 0,
        "wspace": 0,
    }

    cmap = "bwr"
    nb_ticks = 5
    cbar_tick_format = "%.2g"

    fig, axs = plt.subplots(3, n_cond + 1 + 1, figsize=(35, 20), gridspec_kw=gs)

    for i, direction in enumerate(directions):
        ax = axs[i, 0]
        ax.set_axis_off()
        if i == 1:
            continue
        else:
            ax.text(0, 0.5, direction, fontsize=fontsize)

    if mode == "vertex_parcellated":
        lh, rh, vmin, vmax = map_destrieux(lh, rh)
        label = "Feature Importance"
    else:
        vmin, vmax = get_fullrang_minmax(fis_agg[target])
        label = "Feature Importance"

    cnt = 1
    for condition in conditions:
        top = axs[0, cnt]
        middle = axs[1, cnt]
        bottom = axs[2, cnt]

        if mode == "roi_model":
            correct = format_rois(fis_agg[target], "correct", condition)
            make_roi_model_plot(correct, target, top, vmin, vmax)

            incorrect = format_rois(fis_agg[target], "incorrect", condition)
            make_roi_model_plot(incorrect, target, bottom, vmin, vmax)

        else:

            lh_correct = format_for_plotting(lh, "correct", condition)
            rh_correct = format_for_plotting(rh, "correct", condition)
            draw_plot(lh_correct, rh_correct, top, mode, vmin=vmin, vmax=vmax)

            lh_incorrect = format_for_plotting(lh, "incorrect", condition)
            rh_incorrect = format_for_plotting(rh, "incorrect", condition)
            draw_plot(lh_incorrect, rh_incorrect, bottom, mode, vmin=vmin, vmax=vmax)

        middle.set_axis_off()  # make blank space
        top.set_title(condition, fontsize=fontsize)
        cnt += 1

    # plot colorbar
    norm = Normalize(vmin=vmin, vmax=vmax)
    proxy_mappable = ScalarMappable(norm=norm, cmap=cmap)
    ticks = np.linspace(vmin, vmax, nb_ticks)

    right = axs[:, n_cond + 1]

    for ax in right.flat:
        ax.set_axis_off()

    cax, kw = make_axes(right, fraction=0.5, shrink=0.5)
    cbar = fig.colorbar(
        proxy_mappable,
        cax=cax,
        ticks=ticks,
        orientation="vertical",
        format=cbar_tick_format,
        ticklocation="left",
    )
    cbar.set_label(label=label, fontsize=fontsize - 2)

    prefix = "Target: "
    title = target_map[target]

    fig.suptitle(prefix + title, fontsize=fontsize + 2)

    fpath = f"{basepath}/{agg}/{mode}/{model}"
    if not os.path.exists(fpath):
        os.makedirs(fpath)

    plt.savefig(f"{fpath}/{target}.png", dpi=300, bbox_inches="tight")
    plt.close()

    print(f"{mode}/{model}/{target} saved to {fpath}")


def make_colorbar(
    fig, ax, vmin, vmax, cmap, label="Haufe-Transformed Feature Importance"
):

    # plot colorbar
    nb_ticks = 5
    cbar_tick_format = "%.2g"
    norm = Normalize(vmin=vmin, vmax=vmax)
    proxy_mappable = ScalarMappable(norm=norm, cmap=cmap)
    ticks = np.linspace(vmin, vmax, nb_ticks)

    ax.set_axis_off()

    cax, kw = make_axes(ax, fraction=0.5, shrink=0.5)

    fig.colorbar(
        proxy_mappable,
        cax=cax,
        ticks=ticks,
        orientation="vertical",
        format=cbar_tick_format,
        ticklocation="left",
    )


def make_corplot(x, y, fontsize, xlab=None, ylab=None, color="purple", ax=None):

    r, p = pearsonr(x, y)
    if p < 0.001:
        p_str = rf"$R^2 = {r**2:.2f}, p < 0.001$"
    else:
        p_str = rf"$R^2 = {r**2:.2f}, p = {p:.2f}"

    sns.regplot(
        x=x,
        y=y,
        scatter_kws={"alpha": 0.05, "color": color},
        line_kws={"linewidth": 3, "color": color},
        ax=ax,
    )
    ax.set_xlabel(xlab, fontsize=fontsize)
    ax.set_ylabel(ylab, fontsize=fontsize)
    ax.set_xticks([])
    ax.set_yticks([])

    ax.text(0.05, 0.85, p_str, fontsize=fontsize * 0.75, transform=ax.transAxes)


def make_paper_fis_plot(lr_collection, targets, correct, cond, behavior, params):

    def _get_global_minmax(lr_collection, targets):
        vmin, vmax = np.inf, -np.inf
        for target in targets:
            for lr in lr_collection[target]:
                vmin = min(vmin, lr.min(numeric_only=True).min())
                vmax = max(vmax, lr.max(numeric_only=True).max())

        abs_max = max(np.abs(vmin), np.abs(vmax))
        return -abs_max, abs_max

    n_targets = len(targets)
    n_cond = len(cond)
    n_correct = len(correct)

    target_map = params["target_map"]

    nrows = n_targets + 1 + 1
    ncols = 1 + 1 + n_cond + n_correct + 1

    fontsize = 25

    correct_cond = list(product(correct, cond))

    grid = {
        "width_ratios": [10]
        + [2]
        + list(repeat(10, n_cond))
        + list(repeat(10, n_correct))
        + [4],
        "height_ratios": list(repeat(25, n_targets)) + [7] + [25],
        "hspace": 0,
        "wspace": 0,
    }
    cmap = "bwr"

    vmin, vmax = _get_global_minmax(lr_collection, targets)
    color = sns.color_palette("muted")[2]
    fig, axs = plt.subplots(nrows, ncols, figsize=(42, 19), gridspec_kw=grid)

    for i, target in enumerate(targets):
        ax = axs[i, 0]
        ax.set_axis_off()
        ax.text(0, 0.5, target_map[target], fontsize=fontsize)

    # blank space
    for ax in axs[2, :]:
        ax.set_axis_off()

    for ax in axs[:, 1]:
        ax.set_axis_off()

    for j, lab in enumerate(correct_cond):

        correct, condition = lab
        ax = axs[0, j + 2]
        ax.set_title(f"{correct} {condition}".title(), fontsize=fontsize)

        lh, rh = lr_collection[targets[0]]
        lh_plt_t1 = format_for_plotting(lh, correct, condition)
        rh_plt_t1 = format_for_plotting(rh, correct, condition)

        draw_plot(lh_plt_t1, rh_plt_t1, ax, mode="", cmap=cmap, vmin=vmin, vmax=vmax)

        ax = axs[1, j + 2]

        lh, rh = lr_collection[targets[1]]
        lh_plt_t2 = format_for_plotting(lh, correct, condition)
        rh_plt_t2 = format_for_plotting(rh, correct, condition)

        draw_plot(lh_plt_t2, rh_plt_t2, ax, mode="", cmap=cmap, vmin=vmin, vmax=vmax)

        ax = axs[3, j + 2]

        t1 = np.append(lh_plt_t1.values, rh_plt_t1.values).flatten()
        t2 = np.append(lh_plt_t2.values, rh_plt_t2.values).flatten()

        if j == 0:
            ylab = "Features from EEA Model"
            ax.set_title("Feat. Imp. Correlation", fontsize=fontsize, loc="left")
        else:
            ylab = None

        xlab = r"Features from $P_{tf}$ model"

        make_corplot(t1, t2, fontsize, ylab=ylab, xlab=xlab, color=color, ax=ax)

    # plot colorbar
    label = "Haufe-Transformed Feature Importance"
    nb_ticks = 5
    cbar_tick_format = "%.2g"
    norm = Normalize(vmin=vmin, vmax=vmax)
    proxy_mappable = ScalarMappable(norm=norm, cmap=cmap)
    ticks = np.linspace(vmin, vmax, nb_ticks)

    right = axs[0:2, len(correct_cond) + 2]

    for ax in right.flat:
        ax.set_axis_off()

    cax, kw = make_axes(right, fraction=0.5, shrink=0.5)

    cbar = fig.colorbar(
        proxy_mappable,
        cax=cax,
        ticks=ticks,
        orientation="vertical",
        format=cbar_tick_format,
        ticklocation="left",
    )
    cbar.set_label(label=label, fontsize=fontsize - 2)

    ax = axs[nrows - 1, 0]
    make_corplot(
        behavior["tf"].values,
        behavior["EEA"].values,
        fontsize,
        xlab=r"$P_{tf}$",
        ylab=r"$EEA$",
        ax=ax,
    )
    ax.set_title("Parameter Correlation", fontsize=fontsize, loc="left")

    ax = axs[nrows - 1, ncols - 1]
    ax.set_axis_off()
    sns.despine()

    plt.savefig(
        params["plot_output_path"] + "haufe_feature_importance.png",
        dpi=150,
        bbox_inches="tight",
    )
    # plt.show()


def get_global_minmax(lh, rh):

    vmin, vmax = np.inf, -np.inf
    for lr in [lh, rh]:
        vmin = min(vmin, lr.min(numeric_only=True).min())
        vmax = max(vmax, lr.max(numeric_only=True).max())

    abs_max = max(np.abs(vmin), np.abs(vmax))
    return -abs_max, abs_max


def make_supplement_plot(fis, params, title, roi=False):

    targets = list(fis.keys())

    lr_collection = {}
    for target in targets:
        lr_collection[target] = broadcast_to_fsaverage(fis[target])

    conditions = pd.unique(lr_collection[targets[0]][0]["condition"])
    correct = pd.unique(lr_collection[targets[0]][0]["correct"])
    target_map = params["target_map"]

    correct_cond = list(product(correct, conditions))
    correct_cond = [("", "")] + correct_cond + [("", "")]  # pad

    n_targets = len(targets)

    grid = {
        "width_ratios": [8] + list(repeat(10, 4)) + [2],
        "wspace": 0,
    }
    fig, axs = plt.subplots(n_targets, 6, figsize=(35, 4 * n_targets), gridspec_kw=grid)

    cmap = "bwr"
    fontsize = 15

    mode = "vertex_parcellated" if roi else ""

    for i, target in enumerate(targets):
        lh, rh = lr_collection[target]

        if roi:
            lh, rh, vmin, vmax = map_destrieux(lh, rh)
        else:
            vmin, vmax = get_global_minmax(lh, rh)

        ax_row = axs[i, :]
        for (correct, condition), ax in zip(correct_cond, ax_row.flat):

            if ax == ax_row[0]:
                ax.text(0, 0.5, target_map[target], fontsize=fontsize)
                ax.set_axis_off()
            elif ax != ax_row[-1]:
                lh_plt = format_for_plotting(lh, correct, condition)
                rh_plt = format_for_plotting(rh, correct, condition)

                draw_plot(
                    lh_plt, rh_plt, ax, mode=mode, cmap=cmap, vmin=vmin, vmax=vmax
                )
                ax.set_title(f"{correct} {condition}".title(), fontsize=fontsize)
            else:
                make_colorbar(fig, ax, vmin, vmax, cmap, label="")


def gather_fis(fis: list, compare_scopes: bool):
    """Gather feature importance for plotting."""

    best_fis = {}
    avg_fis = {}

    if compare_scopes:
        targets = fis[0][1].keys()

        for target in targets:
            best_fis[target] = pd.concat([f[1][target] for f in fis])
            avg_fis[target] = pd.concat([f[2][target] for f in fis])
    else:
        print(fis)
        targets = fis.keys()

        for target in targets:
            best_fis[target] = pd.concat([f[target] for f in fis])
            avg_fis[target] = pd.concat([f[target] for f in fis])

    return best_fis, avg_fis


def plot_mode(
    fis_agg: dict, target_map: dict, agg: str, mode: str, model: str, basepath: str
):
    """Plot mode.

    Args:
        fis_agg (dict): Feature importance.
        target_map (dict): Target map.
        agg (str): Aggregation.
        mode (str): Mode.
        model (str): Model.
        basepath (str, optional): Base path.
    """

    Parallel(n_jobs=8)(
        delayed(make_collage_plot)(
            fis_agg,
            target,
            target_map,
            basepath=basepath,
            agg=agg,
            mode=mode,
            model=model,
        )
        for target in fis_agg.keys()
    )


def make_fis_plots(
    avg_fis,
    best_fis,
    haufe_fis,
    target_map,
    model="enet",
    basepath="../../data/06_reporting/rdex_prediction/fis_plots",
):
    """Make fis plots.

    Args:
        avg_fis (pd.DataFrame): Avg. feature importance.
        best_fis (pd.DataFrame): Best feature importance.
        target_map (dict): Target map.
        model (str, optional): Model. Defaults to 'enet'.
        basepath (str, optional): Base path.
    """

    # modes = ["vertex", "roi", "roi_model"]
    # modes = ["roi_model"]
    modes = ["vertex", "vertex_parcellated"]
    # modes = ["vertex_parcellated"]

    for mode in modes:
        plot_mode(
            avg_fis,
            target_map,
            mode=mode,
            agg="avg_fis",
            model=model,
            basepath=basepath,
        )
        plot_mode(
            best_fis,
            target_map,
            mode=mode,
            agg="best_fis",
            model=model,
            basepath=basepath,
        )
        plot_mode(
            haufe_fis,
            target_map,
            mode=mode,
            agg="haufe_fis",
            model=model,
            basepath=basepath,
        )


def draw_supplement(mod_name, values, params):

    fpath = values["fpath"]
    title = values["title"]
    roi = values["roi"]

    fis, best_fis, avg_fis, haufe_avg, haufe_fis = pd.read_pickle(fpath)

    make_supplement_plot(haufe_avg, params, title=title, roi=roi)

    fname_out = f"{mod_name}_haufe_fis_supplement.png"
    plt.savefig(params["plot_output_path"] + fname_out, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved {fname_out}")


def produce_supplement_plots(params: dict):

    fpath_ridge = params["model_results_path"] + "vertex_ridge_feature_importance.pkl"
    fpath_lasso = params["model_results_path"] + "vertex_lasso_feature_importance.pkl"
    (params["model_results_path"] + "all_vertex_contrasts_ridge_feature_importance.pkl")

    plot_names = {
        "ridge": {
            "fpath": fpath_ridge,
            "title": "Ridge Regression Feature Importance",
            "roi": False,
        },
        "ridge_roi": {
            "fpath": fpath_ridge,
            "title": "Ridge Regression Feature Importance (Destrieux)",
            "roi": True,
        },
        "lasso": {
            "fpath": fpath_lasso,
            "title": "Lasso Regression Feature Importance",
            "roi": False,
        },
        "lasso_roi": {
            "fpath": fpath_lasso,
            "title": "Lasso Regression Feature Importance (Destrieux)",
            "roi": True,
        },
    }

    Parallel(n_jobs=4)(
        delayed(draw_supplement)(mod_name, values, params)
        for mod_name, values in plot_names.items()
    )


def produce_fis_plot(params: dict, haufe_avg: dict):
    targets = ["EEA", "tf"]
    len(targets)

    lr_collection = {}
    for target in targets:
        lr_collection[target] = broadcast_to_fsaverage(haufe_avg[target])

    behavior = pd.read_csv(params["targets_path"])

    correct = ["correct", "incorrect"]
    cond = ["go", "stop"]

    make_paper_fis_plot(lr_collection, targets, correct, cond, behavior, params)


def produce_effectsize_plot(params: dict, model: str):

    process_map = params["process_map"]
    target_map = params["target_map"]
    color_map = params["color_map"]

    # paper effectsize plot
    (
        pd.read_csv(params["model_results_path"] + f"{model}_models_summary.csv")
        .pipe(relabel_plotting_data, process_map, target_map, color_map)
        .pipe(sort)
        .pipe(make_paper_effectsize_plot, params)
    )


def produce_plots(params: dict, model: str, compare_scopes=False):
    """Produce plots.

    Args:
        params (dict): Parameters.
        model (str): Model.
    """

    params["process_map"]
    params["target_map"]
    params["color_map"]

    fis, best_fis, avg_fis, haufe_avg, haufe_fis = pd.read_pickle(
        params["model_results_path"] + f"{model}_feature_importance.pkl"
    )

    produce_fis_plot(params, haufe_avg)

    # effect compare

    # if compare_scopes:
    #     fis = pd.read_pickle(
    #         params["model_results_path"] + f"{model}_feature_importance.pkl"
    #     )

    #     best_fis, avg_fis = gather_fis(fis, compare_scopes=compare_scopes)

    #     make_fis_plots(avg_fis, best_fis, target_map, model=model)

    # else:
    #     fis, best_fis, avg_fis, haufe_avg, haufe_fis = pd.read_pickle(
    #         params["model_results_path"] + f"{model}_feature_importance.pkl"
    #     )

    #     make_fis_plots(avg_fis, best_fis, haufe_avg, target_map, model=model)
