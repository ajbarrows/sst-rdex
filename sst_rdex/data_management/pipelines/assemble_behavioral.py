import pandas as pd

from abcd_tools.utils.io import load_tabular, save_csv
from abcd_tools.utils.ConfigLoader import load_yaml
from abcd_tools.task.metrics import DPrimeDataset


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
    rdex_map = rdex_map.drop(columns=[v for v in exclude_vars if v in rdex_map.columns])
    rdex_map.insert(1, "eventname", tpt)

    return rdex_map.set_index(["src_subject_id", "eventname"])


def load_nback_behavioral(params: dict) -> pd.DataFrame:
    """Load the NBack behavioral data
    Args:
        params (dict): Parameters loaded from YAML file

    Returns:
        pd.DataFrame: renamed and filtered NBack behavioral data
    """

    nback_behavioral = load_tabular(
        params["nback_behavioral_path"],
        cols=params["nback_rt_vars"],
        timepoints=["baseline_year_1_arm_1"],
    )

    dprime = DPrimeDataset()
    dprime = dprime.load_and_compute(pd.read_csv(params["nback_behavioral_path"]))
    dprime["dprime_avg"] = dprime[["dprime_0back", "dprime_2back"]].mean(axis=1)

    return nback_behavioral.join(dprime, how="left")


def load_nback_rdex(params: dict) -> pd.DataFrame:
    """Load the NBack RDEX MAP estimates

    Args:
        params (dict): Parameters loaded from YAML file

    Returns:
        pd.DataFrame: NBack RDEX MAP estimates
    """

    def split(df):
        tmp = df["parameter"].str.split(".", expand=True)
        tmp = tmp.rename(columns={0: "condition", 1: "parameter"})

        df = df.drop(columns="parameter")
        return pd.concat([df, tmp], axis=1)

    # nback_medians = (
    #     pd.read_csv(params['nback_rdex_median_path'], index_col=0)
    #     .reset_index(names='src_subject_id')
    #     )
    zeroback = pd.read_csv(params["nback_rdex_map_0back"]).set_index("subjectkey")
    twoback = pd.read_csv(params["nback_rdex_map_2back"]).set_index("subjectkey")

    nback_maps = pd.concat([zeroback, twoback], axis=1).reset_index(
        names="src_subject_id"
    )
    nback_maps = (
        nback_maps.melt(
            id_vars="src_subject_id", var_name="parameter", value_name="median"
        )
        .pipe(split)
        .pivot(
            index=["src_subject_id", "condition"], columns="parameter", values="median"
        )
        .reset_index()
        .groupby("src_subject_id")
        .mean()
        .filter("e")
    )

    return nback_maps


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
    save_csv(filtered_behavioral, params["filtered_behavioral_path_sst"])

    rdex_map_no_tf = load_rdex_map(params["rdex_map_notf_path"])
    filtered_behavioral_no_tf = filter_behavioral(rdex_map_no_tf, sst_behavioral)
    save_csv(filtered_behavioral_no_tf, params["filtered_behavioral_no_tf_path_sst"])

    nback_behavioral = load_nback_behavioral(params)
    nback_medians = load_nback_rdex(params)

    nback_behavioral = filter_behavioral(nback_medians, nback_behavioral)
    nback_behavioral = nback_behavioral[nback_behavioral.index.isin(rdex_map.index)]
    print(f"Subjects with nBack and SST behavioral data: {len(nback_behavioral)}")
    save_csv(nback_behavioral, params["filtered_behavioral_path_nback"])


if __name__ == "__main__":
    main()
