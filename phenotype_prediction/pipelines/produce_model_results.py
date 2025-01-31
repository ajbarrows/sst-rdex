import pandas as pd
import numpy as np
import BPt as bp

import matplotlib.pyplot as plt
import seaborn as sns

from abcd_tools.utils.ConfigLoader import load_yaml


def get_full_summary(res: bp.CompareDict) -> pd.DataFrame:
    """Helper to get full summary information from BPt models.

    Args:
        res (bp.CompareDict): Model results.

    Returns:
        pd.DataFrame: Full summary information
    """

    keys = list(res.keys())
    repr_key = keys[0]
    option_keys = [o.key for o in repr_key.options]
    cols = {key: [] for key in option_keys}

    for key in keys:
        for option in key.options:
            cols[option.key].append(option.name)

        evaluator = res[key]

        attr = getattr(evaluator, "scores")
        new_col_names = []
        for key in attr:

            val = attr[key]

            new_col_name = "scores" + "_" + key
            new_col_names.append(new_col_name)

            try:
                cols[new_col_name].append(val)
            except KeyError:
                cols[new_col_name] = [val]

    s = pd.DataFrame.from_dict(cols)
    return s.explode(new_col_names)


def make_phenotype_plot_df(res: bp.CompareDict, params: dict) -> pd.DataFrame:
    """Make phenotype plot dataframe.

    Args:
        res (bp.CompareDict): Model results.
        params (dict): Parameters.

    Returns:
        pd.DataFrame: Phenotype plot dataframe.
    """

    summary = get_full_summary(res)

    item_map = params["phenotype_plot_name_keyed"]
    grouping_map = params["grouping_map"]

    summary = summary.replace(item_map)
    summary = summary.replace(grouping_map)

    tmp = summary["target"].str.split(":", expand=True)
    tmp.columns = ["grouping", "item"]
    summary = pd.concat([summary, tmp], axis=1)

    summary = summary.sort_values(["grouping", "scope", "scores_r2"], ascending=False)

    return summary


def make_phenotype_effectsize_plot(plot_df: pd.DataFrame, params: dict) -> None:
    """Make phenotype effectsize plot.

    Args:
        plot_df (pd.DataFrame): Plot dataframe.
        params (dict): Parameters.
    """

    sns.set(style="whitegrid", font_scale=2)

    g = sns.FacetGrid(plot_df, col="grouping", height=10, sharex=False)
    g.map(sns.barplot, "item", "scores_r2", "scope", palette="viridis")
    g.set_xticklabels(rotation=45, ha="right")
    g.set_titles("{col_name}")
    g.set_axis_labels("", "$R^2$")
    g.add_legend(title="")

    plt.savefig(
        params["phenotype_plot_output_dir"] + "/phenotype_effectsize_plot.png",
        bbox_inches="tight",
        dpi=300,
    )


def gather_phenotype_fis(res: bp.CompareDict, params: dict) -> pd.DataFrame:
    """Gather phenotype feature importance scores.

    Args:
        res (bp.CompareDict): Model results.
        params (dict): Parameters.

    Returns:
        pd.DataFrame: Phenotype feature importance scores.
    """

    item_map = params["phenotype_plot_name_keyed"]
    grouping_map = params["grouping_map"]

    keys = list(res.keys())
    fis = pd.DataFrame()
    for key in keys:
        tmp = res[key].get_fis()
        scope = str(key.options[0]).replace("scope=", "")
        target = str(key.options[1]).replace("target=", "")

        tmp.insert(0, "scope", scope)
        tmp.insert(1, "target", target)

        fis = pd.concat([fis, tmp])

    fis = fis.replace(item_map)
    fis = fis.replace(grouping_map)

    tmp = fis["target"].str.split(":", expand=True)
    tmp.columns = ["grouping", "item"]
    fis = pd.concat([fis, tmp], axis=1)

    return fis


def make_average_fis(fis: pd.DataFrame, params: dict) -> pd.DataFrame:
    """Make average feature importance scores.

    Args:
        fis (pd.DataFrame): Feature importance scores.
        params (dict): Parameters.

    Returns:
        pd.DataFrame: Average feature importance scores.
    """

    covars = params["covariates"]
    target_map = params["target_map"]

    fis = fis.drop(columns=covars)

    fisummary = fis.groupby(["grouping", "scope"]).mean().reset_index()
    fisummary = fisummary.melt(
        id_vars=["grouping", "scope"], var_name="feature", value_name="importance"
    )

    fisummary = fisummary.replace(target_map)
    return fisummary


def make_feat_imp_radar_plot(df, ax, legend=True):
    """Make feature importance radar plot.

    Args:
        df (pd.DataFrame): Dataframe.
        ax (plt.Axes): Axes.
        legend (bool, optional): Legend. Defaults to True.
    """

    df = df.dropna()
    variables = pd.unique(df["feature"])
    N = len(variables)

    categories = df["grouping"].unique()
    colors = ["#1f77b4", "#aec7e8", "#ff7f0e"]

    radians = 2 * np.pi
    angles = [n / float(N) * radians for n in range(N)]
    angles += angles[:1]

    # instantiate plot
    ax.set_xticks(angles[:-1], variables)
    ax.set_rlabel_position(10)

    # plot circle to show 0
    rads = np.arange(0, (2 * np.pi), 0.01)
    zeros = np.zeros(len(rads))
    ax.plot(rads, zeros, "k", alpha=0.5)

    # set grid
    ax.grid(True)
    ax.spines["polar"].set_visible(False)

    for category, color in zip(categories, colors):

        tmp = df[df["grouping"] == category]

        values = tmp["importance"].reset_index(drop=True).values

        values = np.append(values, values[0])

        ax.plot(angles, values, color=color)

    if legend:
        legend_labels = np.insert(categories, 0, "Reference = 0")
        ax.legend(legend_labels, bbox_to_anchor=(0, 1.05))


def phenotype_feat_important_collage(avg_fis: pd.DataFrame, params: dict) -> None:
    """Make phenotype feature importance collage.

    Args:
        avg_fis (pd.DataFrame): Average feature importance scores.
        params (dict): Parameters.
    """

    sns.set_theme()
    sns.set(style="whitegrid", font_scale=1)

    scopes = params["radar_plot_scopes"]

    fig, ax = plt.subplots(
        ncols=len(scopes), figsize=(25, 25), subplot_kw={"projection": "polar"}
    )

    for i, scope in enumerate(scopes):
        legend = True if i == len(scopes) - 1 else False
        make_feat_imp_radar_plot(
            avg_fis[avg_fis["scope"] == scope], ax[i], legend=legend
        )
        ax[i].set_title(scope)

    plt.savefig(
        params["phenotype_plot_output_dir"] + "/phenotype_feat_imp_radar_plot.png",
        bbox_inches="tight",
        dpi=300,
    )


def main():
    params = load_yaml("../parameters.yaml")

    results = pd.read_pickle(
        params["phenotype_output_dir"] + "/phenotype_elastic_results.pkl"
    )
    plot_df = make_phenotype_plot_df(results, params)
    make_phenotype_effectsize_plot(plot_df, params)

    fis = gather_phenotype_fis(results, params)
    avg_fis = make_average_fis(fis, params)
    phenotype_feat_important_collage(avg_fis, params)


if __name__ == "__main__":
    main()
