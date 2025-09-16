import glob
import pandas as pd
import numpy as np
import BPt as bp

from abcd_tools.utils.io import load_tabular
from abcd_tools.utils.ConfigLoader import load_yaml


def load_betas(betas_path: str) -> pd.DataFrame:
    """Load partitioned betas.

    Args:
        betas_path (str): Path to partitioned betas.

    Returns:
        pd.DataFrame: Partitioned betas.
    """
    files = glob.glob(betas_path + "*.parquet")
    df = pd.DataFrame()
    for f in files:
        tmp = pd.read_parquet(f)
        tmp = tmp.dropna(axis=1)
        df = pd.concat([df, tmp], axis=1)
    return df


def strip_suffix(s: str):
    """Strip the first two parts of a string separated by underscores."""
    return "_".join(s.split("_")[2:])


def make_contrast(fpath: str, cond1: tuple, cond2: tuple) -> pd.DataFrame:
    """Make a contrast between two conditions.
    Args:
        fpath (str): Path to the folder containing the betas.
        cond1 (tuple): First condition (string, string).
        cond2 (tuple): Second condition (string, string).
    Returns:
        pd.DataFrame: Contrast between the two conditions.
    """

    files = glob.glob(fpath + "*.parquet")
    df1 = pd.read_parquet([f for f in files if cond1[0] in f][0])
    df2 = pd.read_parquet([f for f in files if cond2[0] in f][0])

    df1 = df1.fillna(0)
    df2 = df2.fillna(0)

    df1.columns = [strip_suffix(col) for col in df1.columns]
    df2.columns = [strip_suffix(col) for col in df2.columns]

    contrast_name = f"{cond1[1]}_{cond2[1]}"
    df = df1 - df2
    df = df.rename(columns={col: f"{contrast_name}_{col}" for col in df.columns})

    return df


def load_contrasts(params: dict) -> pd.DataFrame:
    """Load specific contrasts from processed betas."""
    return pd.concat(
        [
            make_contrast(
                params["sst_betas_path"], ("cs", "correctstop"), ("cg", "correctgo")
            ),
            make_contrast(
                params["sst_betas_path"], ("cs", "correctstop"), ("ig", "incorrectgo")
            ),
            make_contrast(
                params["sst_betas_path"], ("is", "incorrectstop"), ("cs", "correctstop")
            ),
            make_contrast(
                params["sst_betas_path"], ("is", "incorrectstop"), ("ig", "incorrectgo")
            ),
        ],
        axis=1,
    )


def gather_scopes(
    betas: pd.DataFrame, mri_confounds: pd.DataFrame, fpath: str = None, roi=False
) -> dict:
    """Assign names to predictors based on SST condition, plus covariates.

    Args:
        betas (pd.DataFrame): Concatenated betas.
        mri_confounds (pd.DataFrame): MRI confounds.
        fpath (str, optional): Path to save scopes. Defaults to None.

    Returns:
        dict: Scopes
    """
    scopes = {}

    if roi:
        r = set(["_".join(c.rsplit("_", 5)[1:3]) for c in betas.columns])
        unique_regressors = [reg for reg in r if "beta" not in reg]

        for u in unique_regressors:
            scopes[u] = []
            for c in betas.columns:
                if u == "_".join(c.rsplit("_", 5)[1:3]):
                    scopes[u].append(c)

    else:
        unique_regressors = set([c.rsplit("_", 2)[0] for c in betas.columns])

        for u in unique_regressors:
            scopes[u] = []
            for c in betas.columns:
                if u == "_".join(c.split("_", 2)[:2]):
                    scopes[u].append(c)

    scopes["mri_confounds"] = mri_confounds.columns

    if fpath is not None:
        pd.to_pickle(scopes, fpath)
        print(f"Scopes saved to {fpath}")

    return scopes


def make_bpt_dataset(
    betas: pd.DataFrame,
    scopes: dict,
    mri_confounds: pd.DataFrame,
    targets: pd.DataFrame,
    test_split=0.2,
    random_state=42,
    fpath: str = None,
) -> bp.Dataset:
    """Create a BPt dataset from betas, confounds, and targets.

    Args:
        betas (pd.DataFrame): Concatenated betas.
        scopes (dict): Scopes.
        mri_confounds (pd.DataFrame): MRI confounds.
        targets (pd.DataFrame): Behavioral targets.
        test_split (float, optional): Test split. Defaults to 0.2.
        random_state (int, optional): Random state. Defaults to 42.
        fpath (str, optional): Path to save dataset. Defaults to None.

    Returns:
        bp.Dataset: BPt dataset.
    """

    df = pd.concat([betas, mri_confounds, targets], axis=1)
    df = df.dropna(axis=1, how="all").dropna(axis=1, how="all").dropna()

    dataset = bp.Dataset(df, targets=targets.columns.tolist())

    scopes["covariates"] = mri_confounds.columns.tolist()
    for k, v in scopes.items():
        dataset.add_scope(v, k, inplace=True)

    dataset = dataset.auto_detect_categorical()
    dataset = dataset.add_scope("mri_info_deviceserialnumber", "category")
    dataset = dataset.ordinalize("category")

    # deal with possible inf values
    dataset = dataset.replace([np.inf, -np.inf], np.nan)
    dataset = dataset.dropna()

    dataset = dataset.set_test_split(test_split, random_state=random_state)

    if fpath is not None:
        dataset.to_pickle(fpath)
        print(f"Dataset saved to {fpath}")

    return dataset


def main():
    params = load_yaml("../parameters.yaml")

    sst_betas = load_betas(params["sst_betas_path"])
    nback_betas = load_betas(params["nback_betas_path"])

    sst_contrasts = load_contrasts(params)

    mri_confounds_sst = load_tabular(params["mri_confounds_sst_path"])
    mri_confounds_nback = load_tabular(params["mri_confounds_nback_path"])

    gather_scopes(sst_betas, mri_confounds_sst, params["sst_scopes_path"])
    gather_scopes(nback_betas, mri_confounds_nback, params["nback_scopes_path"])

    sst_contrast_scopes = gather_scopes(
        sst_contrasts, mri_confounds_sst, params["sst_contrast_scopes_path"]
    )

    targets = load_tabular(params["targets_path"])
    load_tabular(params["targets_no_tf_path"])
    load_tabular(params["nback_targets_path"])

    # make_bpt_dataset(
    #     sst_betas,
    #     sst_scopes,
    #     mri_confounds_sst,
    #     targets,
    #     fpath=params["sst_dataset_path"],
    # )

    # make_bpt_dataset(
    #     sst_betas,
    #     sst_scopes,
    #     mri_confounds_sst,
    #     targets_no_tf,
    #     fpath=params["sst_dataset_no_tf_path"],
    # )

    # make_bpt_dataset(
    #     nback_betas,
    #     nback_scopes,
    #     mri_confounds_nback,
    #     targets_nback,
    #     fpath=params["nback_dataset_path"],
    # )

    # make_bpt_dataset(
    #     sst_betas,
    #     sst_scopes,
    #     mri_confounds_sst,
    #     targets_nback,
    #     fpath=params["sst_nback_dataset_path"],
    # )

    make_bpt_dataset(
        sst_contrasts,
        sst_contrast_scopes,
        mri_confounds_sst,
        targets,
        fpath=params["sst_contrast_dataset_path"],
    )


if __name__ == "__main__":
    main()
