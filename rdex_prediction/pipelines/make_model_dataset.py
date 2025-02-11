import glob
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
    files = glob.glob(betas_path + "*.parquet")
    df = pd.DataFrame()
    for f in files:
        tmp = pd.read_parquet(f)
        df = pd.concat([df, tmp], axis=1)
    return df


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

    vertex_betas = load_betas(params["betas_path"])
    roi_betas = pd.read_parquet(params["roi_betas_path"])

    mri_confounds = load_tabular(params["mri_confounds_path"])

    vertex_scopes = gather_scopes(vertex_betas, mri_confounds, params["scopes_path"])
    roi_scopes = gather_scopes(
        roi_betas, mri_confounds, params["roi_scopes_path"], roi=True
    )

    targets = load_tabular(params["targets_path"])
    target_no_tf = load_tabular(params["targets_no_tf_path"])

    make_bpt_dataset(
        vertex_betas,
        vertex_scopes,
        mri_confounds,
        targets,
        fpath=params["dataset_path"],
    )

    make_bpt_dataset(
        vertex_betas,
        vertex_scopes,
        mri_confounds,
        target_no_tf,
        fpath=params["dataset_no_tf_path"],
    )

    make_bpt_dataset(
        roi_betas, roi_scopes, mri_confounds, targets, fpath=params["roi_dataset_path"]
    )

    make_bpt_dataset(
        roi_betas,
        roi_scopes,
        mri_confounds,
        target_no_tf,
        fpath=params["roi_dataset_no_tf_path"],
    )


if __name__ == "__main__":
    main()
