"""Data processing utilities for plotting."""

import pandas as pd


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
        .join(avg)
        .sort_values(
            by=["process", "avg_mean", "mean_scores_r2"], ascending=[True, False, False]
        )
        .reset_index()
        .drop(columns=["avg_mean", "avg_std"])
    )
    return df


def absmax(x):
    """Return value with maximum absolute value."""
    return x[x.abs().idxmax()]


def get_fullrang_minmax(series: pd.Series):
    """Get min/max values from series for full range plotting."""
    abs_max = series.abs().max()
    return -abs_max, abs_max


def format_rois(fis_agg: dict, correct, condition):
    """Format ROI data for plotting."""
    fis_collection = []
    for k, v in fis_agg.items():
        lab = k.split("_")[-1]
        tmp = pd.DataFrame(
            {"roi": lab, "fis": v, "correct": correct, "condition": condition}
        )
        fis_collection.append(tmp)

    return pd.concat(fis_collection, axis=0)


def format_for_plotting(
    fis: dict, targets: list, correct: list, conditions: list, include_all=True
):
    """Format feature importance scores for plotting."""
    plot_df_ls = []

    for target in targets:
        for cor in correct:
            for cond in conditions:
                if cond == "all" and not include_all:
                    continue

                lab = f"{target}_{cor}_{cond}"

                if lab not in fis.keys():
                    continue

                plot_df = (
                    pd.DataFrame(fis[lab])
                    .assign(target=target, correct=cor, condition=cond)
                    .reset_index()
                )
                plot_df_ls.append(plot_df)

    return pd.concat(plot_df_ls, axis=0).reset_index()


def gather_fis(fis: list, compare_scopes: bool):
    """Gather feature importance scores."""
    targets = ["beta", "gamma", "tau", "v", "a", "t0"]
    conditions = ["all", "correct_go", "correct_stop", "incorrect_go", "incorrect_stop"]
    correct = ["correct", "notf"]

    if compare_scopes:
        fis_agg = {}
        for f in fis:
            for k, v in f.items():
                fis_agg[k] = v
    else:
        fis_agg = fis[0]

    plot_df = format_for_plotting(fis_agg, targets, correct, conditions)

    return plot_df
