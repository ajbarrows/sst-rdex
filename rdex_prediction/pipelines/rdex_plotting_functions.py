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
        .sort_values(by=["process", "avg_mean"], ascending=[True, False])
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

    hatches = ["", "/", "-", "X", "O"]

    fig, ax = plt.subplots(figsize=(15, 5))

    # greypallete = np.repeat('lightgrey', len(df))

    n_scopes = len(df["scope"].drop_duplicates())
    greypallete = list(np.repeat("lightgrey", n_scopes))

    order = df["target"].drop_duplicates()

    g = sns.barplot(
        x="target",
        y="mean_scores_r2",
        hue="scope",
        data=df,
        palette=greypallete,
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
        dodge=0.5,
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

    if model == "ridge":
        model = "Ridge Regression"
    elif model == "elastic":
        model = "Elastic Net Regression"

    title = title + f"\n{model}"
    fig.subplots_adjust(top=0.9)
    fig.suptitle(title)

    fpath = f"{fpath}.png"
    plt.savefig(fpath, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Plot saved to {fpath}")


def broadcast_to_fsaverage(fis_agg: pd.Series, n_vertices=10242 + 1) -> pd.DataFrame:
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
    fis.columns = pd.to_numeric(fis.columns).sort_values()

    # need to insert blank columns for missing vertices
    vertex_names = [*range(1, n_vertices)]
    null_df = pd.DataFrame(np.nan, columns=vertex_names, index=fis.index)
    null_df = null_df.drop(columns=fis.columns)

    df = fis.join(null_df, how="outer")
    lh, rh = _split_hemisphere(df)

    return lh, rh


def load_destrieux_atlas():
    """Load Destrieux atlas."""
    atlas = fetch_atlas_surf_destrieux()
    return atlas


def map_destrieux(lh: pd.DataFrame, rh: pd.DataFrame, prefix: str = "") -> pd.DataFrame:
    """Map Destrieux atlas.

    Args:
        lh (pd.DataFrame): Left hemisphere.
        rh (pd.DataFrame): Right hemisphere.
        prefix (str, optional): Prefix. Defaults to ''.

    Returns:
        pd.DataFrame: Mapped dataframe.
    """

    dest = load_destrieux_atlas()

    idx = ["correct", "condition"]
    lh = lh.set_index(idx)
    rh = rh.set_index(idx)

    lh_mapped = map_hemisphere(
        lh, mapping=dest["map_left"], labels=dest["labels"], prefix=prefix, suffix=".lh"
    )
    rh_mapped = map_hemisphere(
        rh,
        mapping=dest["map_right"],
        labels=dest["labels"],
        prefix=prefix,
        suffix=".rh",
    )

    lh_mapped.index = lh.index
    rh_mapped.index = rh.index

    df = pd.concat([lh_mapped, rh_mapped], axis=1)
    vmin, vmax = get_fullrang_minmax(df)

    return lh_mapped.reset_index(), rh_mapped.reset_index(), vmin, vmax


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

    if mode == "roi":
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
    """Make collage plot.

    Args:
        fis_agg (dict): Feature importance.
        target (str): Target.
        target_map (dict): Target map.
        basepath (str, optional): Base path.
        agg (str, optional): Aggregation. Defaults to 'avg_fis'.
        mode (str, optional): Mode. Defaults to 'vertex'.
        model (str, optional): Model. Defaults to 'enet'.
        fontsize (int, optional): Font size. Defaults to 25.
    """

    def _format_for_plotting(
        fis: pd.DataFrame, correct: str, condition: str
    ) -> pd.DataFrame:
        """Limit to condition."""
        tmp = fis[(fis["correct"] == correct) & (fis["condition"] == condition)]
        tmp = tmp.drop(columns=["correct", "condition"])
        tmp[np.isnan(tmp)] = 0
        return tmp

    lh, rh = broadcast_to_fsaverage(fis_agg[target])

    conditions = pd.unique(lh["condition"])
    n_cond = conditions.shape[0]
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

    if mode == "roi":
        lh, rh, vmin, vmax = map_destrieux(lh, rh)
    else:
        vmin, vmax = get_fullrang_minmax(fis_agg[target])

    cnt = 1
    for condition in conditions:
        top = axs[0, cnt]
        middle = axs[1, cnt]
        bottom = axs[2, cnt]

        lh_correct = _format_for_plotting(lh, "correct", condition)
        rh_correct = _format_for_plotting(rh, "correct", condition)
        draw_plot(lh_correct, rh_correct, top, mode, vmin=vmin, vmax=vmax)

        middle.set_axis_off()  # make blank space

        lh_incorrect = _format_for_plotting(lh, "incorrect", condition)
        rh_incorrect = _format_for_plotting(rh, "incorrect", condition)
        draw_plot(lh_incorrect, rh_incorrect, bottom, mode, vmin=vmin, vmax=vmax)

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

    cbar.set_label(label="Avg. Feature Imp.", fontsize=fontsize - 2)

    prefix = "Target: "
    title = target_map[target]

    fig.suptitle(prefix + title, fontsize=fontsize + 2)

    fpath = f"{basepath}/{agg}/{mode}/{model}"
    if not os.path.exists(fpath):
        os.makedirs(fpath)

    plt.savefig(f"{fpath}/{target}.png", dpi=300, bbox_inches="tight")
    plt.close()


def gather_fis(fis: list):
    """Gather feature importance for plotting."""

    targets = fis[0][1].keys()
    best_fis = {}
    avg_fis = {}

    for target in targets:
        best_fis[target] = pd.concat([f[1][target] for f in fis])
        avg_fis[target] = pd.concat([f[2][target] for f in fis])

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

    modes = ["vertex", "roi"]

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


def produce_plots(params: dict, model: str):
    """Produce plots.

    Args:
        params (dict): Parameters.
        model (str): Model.
    """

    process_map = params["process_map"]
    target_map = params["target_map"]
    color_map = params["color_map"]

    summary = pd.read_csv(params["model_results_path"] + f"{model}_models_summary.csv")
    summary = relabel_plotting_data(summary, process_map, target_map, color_map)
    summary = sort(summary)

    make_effect_compare_plot(
        summary,
        model,
        params["effectsize_plot_title"],
        params["plot_output_path"] + f"{model}_effectsize_plot",
    )

    fis = pd.read_pickle(
        params["model_results_path"] + f"{model}_feature_importance.pkl"
    )
    best_fis, avg_fis = gather_fis(fis)

    make_fis_plots(avg_fis, best_fis, target_map, model=model)
