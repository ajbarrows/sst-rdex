import pandas as pd

from abcd_tools.utils.io import load_tabular, save_csv
from abcd_tools.utils.ConfigLoader import load_yaml
from abcd_tools.image.preprocess import compute_average_betas


def load_mri_qc(mri_qc_path: str) -> pd.DataFrame:
    mri_qc = load_tabular(mri_qc_path, cols=["imgincl_sst_include"])
    return mri_qc[mri_qc["imgincl_sst_include"] == 1]


def load_degrees_of_freedom(r1_fpath: str, r2_fpath: str) -> pd.DataFrame:
    """Load censored frame information for run averaging.

    Args:
        r1_fpath (str): Filepath to run 1 info
        r2_fpath (str): Filepath to run 2 info

    Returns:
        pd.DataFrame: DOFs for runs 1 and 2
    """
    r1_dof = load_tabular(r1_fpath, cols=["tfmri_sstr1_beta_dof"])
    load_tabular(r2_fpath, cols=["tfmri_sstr2_beta_dof"])

    return pd.concat([r1_dof, r1_dof], axis=1)


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


def combine_betas(
    sst_conditions: list,
    hemispheres: list,
    dof: pd.DataFrame,
    beta_input_dir: str,
    beta_output_dir: str,
    vol_info_path: str,
) -> None:
    """Combine betas for SST conditions

    Args:
        sst_conditions (list): SST conditions
        hemispheres (list): Hemispheres
        dof (pd.DataFrame): Degrees of freedom
        beta_input_dir (str): Directory containing beta data
        beta_output_dir (str): Directory to save combined betas
        vol_info_path (str): Path to volume information

    Returns:
        None
    """

    vol_info = pd.read_parquet(vol_info_path)

    for condition in sst_conditions:
        betas = {}
        for hemi in hemispheres:
            run1_fpath = f"{beta_input_dir}sst_{condition}_beta_r01_{hemi}.parquet"
            run2_fpath = f"{beta_input_dir}sst_{condition}_beta_r02_{hemi}.parquet"

            run1 = pd.read_parquet(run1_fpath)
            run2 = pd.read_parquet(run2_fpath)

            avg_betas = compute_average_betas(run1, run2, vol_info, dof)

            betas[hemi] = avg_betas

        betas_df = concatenate_hemispheres(betas["lh"], betas["rh"])

        betas_df.to_parquet(f"{beta_output_dir}average_betas_{condition}.parquet")


def filter_avg_betas(
    mri_qc_df: pd.DataFrame,
    sst_conditions: list,
    filtered_behavioral_path: str,
    beta_output_dir: str,
    processed_beta_dir: str,
) -> pd.DataFrame:
    """Filter average betas based on QC data

    Args:
        mri_qc_df (pd.DataFrame): MRI QC data
        sst_conditions (list): SST conditions
        beta_output_dir (str): Path to average betas
        processed_beta_dir (str): Path to processed betas

    Returns:
        None
    """

    # load targets
    filtered_behavioral = load_tabular(filtered_behavioral_path)
    n_targets = filtered_behavioral.shape[0]

    for condition in sst_conditions:
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
    motion_path: str, scanner_path: str, timepoints: list
) -> pd.DataFrame:
    motion = load_tabular(
        motion_path, cols=["iqc_sst_all_mean_motion"], timepoints=timepoints
    )
    scanner = load_tabular(
        scanner_path, cols=["mri_info_deviceserialnumber"], timepoints=timepoints
    )

    return pd.concat([motion, scanner], axis=1)


def main():
    params = load_yaml("../parameters.yaml")

    dof = load_degrees_of_freedom(params["mri_r1_dof_path"], params["mri_r2_dof_path"])
    mri_qc_df = load_mri_qc(params["mri_qc_path"])
    combine_betas(
        params["sst_conditions"],
        params["hemispheres"],
        dof,
        params["beta_input_dir"],
        params["beta_output_dir"],
        params["vol_info_path"],
    )

    filter_avg_betas(
        mri_qc_df,
        params["sst_conditions"],
        params["filtered_behavioral_path"],
        params["beta_output_dir"],
        params["processed_beta_dir"],
    )

    mri_confounds = load_mri_confounds(
        params["motion_path"], params["scanner_path"], params["timepoints"]
    )
    save_csv(mri_confounds, params["mri_confounds_output_dir"] + "mri_confounds.csv")


if __name__ == "__main__":
    main()
