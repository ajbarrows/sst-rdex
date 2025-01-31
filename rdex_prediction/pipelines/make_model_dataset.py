import os
import pandas as pd
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
    files = os.listdir(betas_path)
    df = pd.DataFrame()
    for f in files:
        tmp = pd.read_parquet(betas_path + f)
        df = pd.concat([df, tmp], axis=1)
    return df


def gather_scopes(
    betas: pd.DataFrame, mri_confounds: pd.DataFrame, fpath: str = None
) -> dict:
    """Assign names to predictors based on SST condition, plus covariates.

    Args:
        betas (pd.DataFrame): Concatenated betas.
        mri_confounds (pd.DataFrame): MRI confounds.
        fpath (str, optional): Path to save scopes. Defaults to None.

    Returns:
        dict: Scopes
    """
    unique_regressors = set([c.rsplit("_", 2)[0] for c in betas.columns])
    scopes = {}

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

    scopes["covariates"] = mri_confounds.columns.tolist()

    df = pd.concat([betas, mri_confounds, targets], axis=1)
    dataset = bp.Dataset(df, targets=targets.columns.tolist())

    for k, v in scopes.items():
        dataset.add_scope(v, k, inplace=True)

    dataset = dataset.auto_detect_categorical()
    dataset = dataset.add_scope("mri_info_deviceserialnumber", "category")
    dataset = dataset.ordinalize("category")

    dataset = dataset.dropna()

    dataset = dataset.set_test_split(test_split, random_state=random_state)

    if fpath is not None:
        dataset.to_pickle(fpath)
        print(f"Dataset saved to {fpath}")

    return dataset


def main():
    params = load_yaml("../parameters.yaml")
    betas = load_betas(params["betas_path"])
    mri_confounds = load_tabular(params["mri_confounds_path"])
    targets = load_tabular(params["targets_path"])
    scopes = gather_scopes(betas, mri_confounds, params["scopes_path"])
    make_bpt_dataset(
        betas, scopes, mri_confounds, targets, fpath=params["dataset_path"]
    )


if __name__ == "__main__":
    main()
