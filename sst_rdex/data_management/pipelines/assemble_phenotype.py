import pandas as pd
import numpy as np
import glob

from abcd_tools.utils.ConfigLoader import load_yaml
from abcd_tools.utils.io import load_tabular, save_csv


def get_fpaths(keywords: list, base_dir: str) -> list:
    """Get list of file paths using keywords for ABCD behaviora data.

    Args:
        keywords (list): List of keywords to search for in file names.
        base_dir (str): Base directory to search for files.

    Returns:
        list: List of file paths that contain the keywords.
    """
    fpaths = []
    for k in keywords:
        fpaths.extend(glob.glob(base_dir + f"**/*{k}*.csv", recursive=True))
    return fpaths


def assemble_phenotype_vars(phenotype_vars: dict) -> list:
    """Parse dictionary of variable names from parameters

    Args:
        phenotype_vars (dict): dictionary of variable names

    Returns:
        list: list of variable names
    """
    vars = []
    for v in phenotype_vars.values():
        vars.extend(list(v))
    return vars


def load_phenotypes(fpaths: list, timepoints: list, vars: list) -> pd.DataFrame:
    """Assemble phenotypes from multiple files into a single DataFrame

    Args:
        fpaths (list): List of file paths
        timepoints (list): Timepoints to subset
        vars (list): Variables to keep.

    Returns:
        pd.DataFrame: Concatenated DataFrame
    """
    phenotypes = pd.DataFrame()
    for fpath in fpaths:
        tmp = load_tabular(fpath, timepoints=timepoints, cols=vars)
        phenotypes = pd.concat([phenotypes, tmp], axis=1)
    return phenotypes


def load_covariates(params: dict) -> pd.DataFrame:
    """Load covariates from file

    Args:
        params (dict): Parameters dictionary

    Returns:
        pd.DataFrame: Covariates DataFrame
    """
    demo = load_tabular(
        params["demo_fpath"], cols=params["demo_vars"], timepoints=params["timepoints"]
    )
    general = load_tabular(
        params["general_fpath"],
        cols=params["general_vars"],
        timepoints=params["timepoints"],
    )
    covars = pd.concat([demo, general], axis=1)

    covars.replace({777: np.nan, 999.0: np.nan}, inplace=True)
    return covars


def join_covariates(phenotypes: pd.DataFrame, covariates: pd.DataFrame) -> pd.DataFrame:
    """Join phenotypes with covariates

    Args:
        phenotypes (pd.DataFrame): Phenotypes DataFrame
        covariates (pd.DataFrame): Covariates DataFrame

    Returns:
        pd.DataFrame: Joined DataFrame
    """
    return phenotypes.join(covariates)


def main():
    # Load parameters
    params = load_yaml("../parameters.yaml")
    phenotype_vars = assemble_phenotype_vars(params["phenotype_vars"])
    fpaths = get_fpaths(params["phenotype_keywords"], params["phenotype_base_dir"])

    # Load phenotypes
    phenotypes = load_phenotypes(fpaths, params["timepoints"], phenotype_vars)

    # Load covariates
    covariates = load_covariates(params)

    # Join phenotypes with covariates
    phenotypes = join_covariates(phenotypes, covariates)

    # Save phenotypes
    save_csv(phenotypes, params["phenotype_output_dir"] + "phenotypes.csv")


if __name__ == "__main__":
    main()
