import pandas as pd

import glob

from scipy.io import loadmat

from abcd_tools.utils.io import load_tabular, apply_nda_names, save_csv
from abcd_tools.utils.ConfigLoader import load_yaml
from abcd_tools.image.preprocess import compute_average_betas


def load_mri_qc(mri_qc_path: str, task: str) -> pd.DataFrame:
    qc_var = "imgincl_" + task + "_include"
    mri_qc = load_tabular(mri_qc_path, cols=[qc_var])
    return mri_qc[mri_qc[qc_var] == 1]


def load_degrees_of_freedom(r1_fpath: str, r2_fpath: str, task: str) -> pd.DataFrame:
    """Load censored frame information for run averaging.

    Args:
        r1_fpath (str): Filepath to run 1 info
        r2_fpath (str): Filepath to run 2 info

    Returns:
        pd.DataFrame: DOFs for runs 1 and 2
    """

    if task == "sst":
        run1_var = "tfmri_sstr1_beta_dof"
        run2_var = "tfmri_sstr2_beta_dof"
    elif task == "nback":
        run1_var = "tfmri_nback_run1_beta_dof"
        run2_var = "tfmri_nback_r2_beta_dof"

    r1_dof = load_tabular(r1_fpath, cols=[run1_var])
    r2_dof = load_tabular(r2_fpath, cols=[run2_var])

    return pd.concat([r1_dof, r2_dof], axis=1)


def concatenate_hemispheres(lh: pd.DataFrame, rh: pd.DataFrame) -> pd.DataFrame:
    """Concatenate left and right hemisphere dataframes

    Args:
        lh (pd.DataFrame): Left hemisphere data
        rh (pd.DataFrame): Right hemisphere data

    Returns:
        pd.DataFrame: Concatenated data
    """
    lh.columns = [c + "_lh" for c in lh.columns]
    rh.columns = [c + "_rh" for c in rh.columns]
    return pd.concat([lh, rh], axis=1)


def parse_vol_info(vol_info: pd.DataFrame) -> pd.DataFrame:

    TPT_MAP = {
        "baseline": "baseline_year_1_arm_1",
        "2year": "2_year_follow_up_y_arm_1",
        "4year": "4_year_follow_up_y_arm_1",
        "6year": "6_year_follow_up_y_arm_1",
    }

    tmp = vol_info.iloc[:, 0].str.split("_", expand=True)[[2, 3]]
    tmp.columns = ["src_subject_id", "eventname"]
    tmp["src_subject_id"] = "NDAR_" + tmp["src_subject_id"]
    tmp["eventname"] = tmp["eventname"].map(TPT_MAP)

    return tmp


def combine_betas(
    conditions: dict,
    hemispheres: list,
    dof: pd.DataFrame,
    task: str,
    beta_input_dir: str,
    beta_output_dir: str,
    vol_info_path: str,
    release: str = "r5",
    rem_outliers: bool = True,
    normalize: bool = True,
) -> None:
    """Combine betas for task conditions

    Args:
        conditions (dict): task conditions
        hemispheres (list): Hemispheres
        beta_input_dir (str): Directory containing beta data
        beta_output_dir (str): Directory to save combined betas
        vol_info_path (str): Path to volume information

    Returns:
        None
    """

    vol_info = pd.read_parquet(vol_info_path)
    if release == "r6":
        vol_info = parse_vol_info(vol_info)

    task_map = {"sst": "SST", "nback": "nBack", "mid": "MID"}

    for condition in conditions.keys():
        betas = {}
        for hemi in hemispheres:

            if release == "r5":
                task_read = task_map[task]
                run1_fpath = f"{beta_input_dir}{task_read}_1_{conditions[condition]}\
                    -{hemi}.parquet"
                run2_fpath = f"{beta_input_dir}{task_read}_2_{conditions[condition]}\
                    -{hemi}.parquet"
            elif release == "r6":
                run1_fpath = (
                    f"{beta_input_dir}{task}_{condition}_beta_r01_{hemi}.parquet"
                )
                run2_fpath = (
                    f"{beta_input_dir}{task}_{condition}_beta_r02_{hemi}.parquet"
                )

            run1 = pd.read_parquet(run1_fpath)
            run2 = pd.read_parquet(run2_fpath)

            name = conditions[condition]
            avg_betas = compute_average_betas(
                run1,
                run2,
                vol_info,
                dof,
                name=name,
                release=release,
                rem_outliers=rem_outliers,
                normalize=normalize,
            )

            betas[hemi] = avg_betas

        betas_df = concatenate_hemispheres(betas["lh"], betas["rh"])

        betas_df.to_parquet(f"{beta_output_dir}average_betas_{condition}.parquet")


def filter_avg_betas(
    mri_qc_df: pd.DataFrame,
    conditions: list,
    filtered_behavioral_path: str,
    beta_output_dir: str,
    processed_beta_dir: str,
) -> pd.DataFrame:
    """Filter average betas based on QC data

    Args:
        mri_qc_df (pd.DataFrame): MRI QC data
        conditions (list): onditions
        beta_output_dir (str): Path to average betas
        processed_beta_dir (str): Path to processed betas

    Returns:
        None
    """

    # load targets
    filtered_behavioral = load_tabular(filtered_behavioral_path)
    n_targets = filtered_behavioral.shape[0]

    for condition in conditions:
        avg_betas_fpath = f"{beta_output_dir}average_betas_{condition}.parquet"
        avg_betas = pd.read_parquet(avg_betas_fpath)

        # limit to available targets
        avg_betas = avg_betas[avg_betas.index.isin(filtered_behavioral.index)]

        print(
            f"{avg_betas.shape[0]} of {n_targets} subjects had MRI data for {condition}"
        )

        nrows_before = avg_betas.shape[0]
        avg_betas = avg_betas[avg_betas.index.isin(mri_qc_df.index)]

        diff = nrows_before - avg_betas.shape[0]

        print(f"{diff} failed MRI QC for {condition}")

        avg_betas.to_parquet(f"{processed_beta_dir}processed_betas_{condition}.parquet")

        del avg_betas


def load_mri_confounds(
    motion_path: str,
    scanner_path: str,
    timepoints: list,
    task: str,
    manufacturer_filter: str = None,
) -> pd.DataFrame:
    motion = load_tabular(
        motion_path, cols=[f"iqc_{task}_all_mean_motion"], timepoints=timepoints
    )

    if manufacturer_filter is not None:
        scanner = load_tabular(
            scanner_path,
            cols=["mri_info_deviceserialnumber", "mri_info_manufacturer"],
            timepoints=timepoints,
        )
        scanner = scanner[scanner["mri_info_manufacturer"] != manufacturer_filter]
        scanner = scanner.drop(columns=["mri_info_manufacturer"])
    else:
        scanner = load_tabular(
            scanner_path, cols=["mri_info_deviceserialnumber"], timepoints=timepoints
        )

    return pd.concat([motion, scanner], axis=1).dropna()


def load_contrasts(
    fpath, dict_path="../../data/01_raw/tabular/abcd_5-1_dictionary.csv"
):
    contrasts = load_tabular(fpath)
    contrasts = apply_nda_names(contrasts, dict_path)
    return contrasts


def strip_names(
    df: pd.DataFrame,
    names=list,
    idx: list = ["src_subject_id", "eventname", "condition"],
    set_index=True,
    reset_index=True,
) -> pd.DataFrame:
    """Remove strings from column names.

    Args:
        df (pd.DataFrame): Dataframe whose columns will be modified.
        names (list): List of strings to remove from column names.
        idx (list): Index columns. Defaults to
            ['src_subject_id', 'eventname', 'condition'].

    Returns:
        pd.DataFrame: Dataframe with  modified column names.
    """
    if set_index:
        df = df.set_index(idx)
    columns = [c.replace(n, "") for n in names for c in df.columns if n in c]

    if len(columns) == 0:
        pass
    else:
        df.columns = columns

    if reset_index:
        df = df.reset_index()
    return df


def drop_cols(df: pd.DataFrame, drop_strings: list) -> pd.DataFrame:
    """Helper function to drop columns containing a given string.

    Args:
        df (pd.DataFrame): Dataframe to filter.
        drop_strings (list): List of strings to drop.

    Returns:
        pd.DataFrame: Filtered dataframe.
    """
    cols = [c for c in df.columns if not any([s in c for s in drop_strings])]
    return df[cols]


def recover_conditions(df1, df2, names1, names2, newname):
    idx = ["src_subject_id", "eventname"]
    df1 = strip_names(df1, names1, set_index=False).set_index(idx)
    df2 = strip_names(df2, names2, set_index=False).set_index(idx)

    df1, df2 = df1.align(df2, join="inner", axis=1)
    df = df1 + df2

    df = drop_cols(df, ["run1", "run2", "sem"])
    df.columns = [c.replace("all", "all_" + newname) for c in df.columns]

    return df


def make_roi_dataset(params: dict) -> pd.DataFrame:
    """Create a dataset with ROI data.

    Args:
        params (dict): Dictionary with parameters.

    Returns:
        pd.DataFrame: ROI dataset.
    """

    base_path = params["roi_base_path"]

    csvcg_dst = load_contrasts(base_path + "mri_y_tfmr_sst_csvcg_dst.csv")
    isvcg_dst = load_contrasts(base_path + "mri_y_tfmr_sst_isvcg_dst.csv")
    igvcg_dst = load_contrasts(base_path + "mri_y_tfmr_sst_igvcg_dst.csv")
    cgvfx_dst = load_contrasts(base_path + "mri_y_tfmr_sst_cgvfx_dst.csv")

    sst_csvcg_names = params["sst_csvcg_names"]
    sst_cgvfx_names = params["sst_cgvfx_names"]
    sst_isvcg_names = params["sst_isvcg_names"]
    sst_igvcg_names = params["sst_igvcg_names"]

    correct_stop = recover_conditions(
        csvcg_dst, cgvfx_dst, sst_csvcg_names, sst_cgvfx_names, "correct_stop"
    )
    incorrect_stop = recover_conditions(
        isvcg_dst, cgvfx_dst, sst_isvcg_names, sst_cgvfx_names, "incorrect_stop"
    )
    incorrect_go = recover_conditions(
        igvcg_dst, cgvfx_dst, sst_igvcg_names, sst_cgvfx_names, "incorrect_go"
    )

    cgvfx_dst = drop_cols(cgvfx_dst, ["run1", "run2", "sem"])
    cgvfx_dst.columns = [
        c.replace("all", "all_" + "correct_go") for c in cgvfx_dst.columns
    ]

    df = pd.concat([correct_stop, incorrect_stop, incorrect_go, cgvfx_dst], axis=1)
    return df


def filter_rois(roi_betas: pd.DataFrame, params: dict) -> pd.DataFrame:
    """Filter ROI dataset.

    Args:
        roi_betas (pd.DataFrame): ROI dataset.
        params (dict): Dictionary with parameters.

    Returns:
        pd.DataFrame: Filtered ROI dataset.
    """

    filtered_behavioral = load_tabular(params["filtered_behavioral_path"])
    mri_qc = load_tabular(params["mri_qc_path"])

    roi_betas = roi_betas[roi_betas.index.isin(filtered_behavioral.index)]
    roi_betas = roi_betas[roi_betas.index.isin(mri_qc.index)]

    return roi_betas


def load_betas(params: dict) -> pd.DataFrame:
    """
    Load the betas from the R5 release

    Args:
        params: dictionary of parameters

    Returns:
        DataFrame with the betas

    """

    # load subject ids
    pguids_path = params["mri_path_r5"] + "sst_regressors_pguids.csv"
    pguids = pd.read_csv(pguids_path)
    pguids = ["NDAR_" + c for c in pguids.columns]

    # load betas
    file_list = glob.glob(params["mri_path_r5"] + "/*.mat")

    names = [f.split("/")[-1] for f in file_list]
    parts = [n.rsplit("_", 4) for n in names]

    conditions = [p[0] for p in parts]
    hemis = [p[-1].rstrip(".mat") for p in parts]

    df = pd.DataFrame(index=pguids)

    for file, condition, hemi in zip(file_list, conditions, hemis):

        data = loadmat(file)["merged_" + hemi]
        data = pd.DataFrame(data.T, index=pguids)
        data.columns = [condition + "_" + hemi + "_" + str(i) for i in data.columns]

        df = pd.concat([df, data], axis=1)

    df = df.reset_index(names="src_subject_id")
    df.insert(1, "eventname", "baseline_year_1_arm_1")

    return df


def main():
    params = load_yaml("../parameters.yaml")

    # for release in ["r5", "r6"]:
    # release = "r5"
    release = "r6"

    # for task in ['sst', 'nback']:
    # for task in ['sst']:
    for task in ["nback"]:
        dof = load_degrees_of_freedom(
            params[f"{task}_r1_dof_path_{release}"],
            params[f"{task}_r2_dof_path_{release}"],
            task,
        )
        mri_qc_df = load_mri_qc(params["mri_qc_path"], task)

        conditions = params[f"{task}_conditions"]

        combine_betas(
            conditions,
            params["hemispheres"],
            dof,
            task,
            params[f"beta_input_dir_{release}"],
            params[f"beta_output_dir_{release}"],
            params[f"vol_info_path_{release}"],
            release=release,
        )

        filter_avg_betas(
            mri_qc_df,
            conditions.keys(),
            params[f"filtered_behavioral_path_{task}"],
            params[f"beta_output_dir_{release}"],
            params[f"processed_beta_dir_{release}"] + f"{task}/",
        )

        mri_confounds = load_mri_confounds(
            params["motion_path"], params["scanner_path"], params["timepoints"], task
        )

        save_csv(
            mri_confounds,
            params["mri_confounds_output_dir"] + f"mri_confounds_{task}.csv",
        )

    # roi_betas = make_roi_dataset(params)
    # roi_betas = filter_rois(roi_betas, params)
    # roi_betas.to_parquet(params["roi_betas_output_dir"])


if __name__ == "__main__":
    main()
