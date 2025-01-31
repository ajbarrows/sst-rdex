import pandas as pd

from abcd_tools.utils.io import load_tabular, save_csv
from abcd_tools.utils.ConfigLoader import load_yaml


def load_sst_behavioral(sst_behavioral_path: str, columns: dict) -> pd.DataFrame:
    """Load the SST behavioral data

    Args:
        sst_behavioral_path (str): Path to NDA SST behavioral data
        columns (dict): Parameters loaded from YAML file

    Returns:
        pd.DataFrame: renamed and filtered SST behavioral data
    """
    sst_behavioral = load_tabular(
        sst_behavioral_path, cols=columns, timepoints=["baseline_year_1_arm_1"]
    )
    return sst_behavioral


def load_rdex_map(
    rdex_map_path: str,
    tpt="baseline_year_1_arm_1",
    exclude_vars: list = ["tf.natural", "gf.natural"],
) -> pd.DataFrame:
    """Load RDEX MAP estimates

    Args:
        rdex_map_path (str): Path to exported RDEX MAP estimates
        tpt (str, optional): Impose timepoint. Defaults to 'baseline_year_1_arm_1'.

    Returns:
        pd.DataFrame: RDEX MAP estimates with index set to src_subject_id and eventname
    """
    rdex_map = pd.read_csv(rdex_map_path)
    rdex_map = rdex_map.rename(columns={"subjectkey": "src_subject_id"})
    rdex_map = rdex_map.drop(columns=exclude_vars)
    rdex_map.insert(1, "eventname", tpt)

    return rdex_map.set_index(["src_subject_id", "eventname"])


def filter_behavioral(
    rdex_map: pd.DataFrame, sst_behavioral: pd.DataFrame
) -> pd.DataFrame:
    """Filter SST behavioral data to subjects with RDEX MAP estimates

    Args:
        rdex_map (pd.DataFrame): RDEX MAP estimates
        sst_behavioral (pd.DataFrame): SST behavioral data

    Returns:
        pd.DataFrame: SST behavioral data for subjects with RDEX MAP estimates
    """

    rdex_joined = rdex_map.join(sst_behavioral, how="inner")
    beh_cols = list(sst_behavioral.columns)

    print(f"Subjects with RDEX MAP estimates: {len(rdex_map)}")
    print(f"{len(rdex_map) - len(rdex_joined)} subs missing {beh_cols}")
    print(f"Subjects with complete behavioral data: {len(rdex_joined)}")

    return rdex_joined


def main():
    params = load_yaml("../parameters.yaml")
    sst_behavioral = load_sst_behavioral(
        params["sst_behavioral_path"], params["beh_rt_vars"]
    )
    rdex_map = load_rdex_map(params["rdex_map_path"])
    filtered_behavioral = filter_behavioral(rdex_map, sst_behavioral)
    save_csv(filtered_behavioral, params["filtered_behavioral_path"])
