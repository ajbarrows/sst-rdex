import glob

import pandas as pd
import numpy as np

from joblib import Parallel, delayed

import BPt as bp
from abcd_tools.utils.ConfigLoader import load_yaml

from rdex_plotting_functions import produce_plots


def join_test_prediction(
    results_summary: pd.DataFrame, test_prediction: pd.DataFrame
) -> pd.DataFrame:
    """Join test predictions with results summary.

    Args:
        results_summary (pd.DataFrame): Results summary.
        test_prediction (pd.DataFrame): Test predictions.

    Returns:
        pd.DataFrame: Joined dataframes.
    """
    idx = ["target", "scope"]
    return results_summary.join(test_prediction.set_index(idx), on=idx, how="left")


def get_test_prediction(results, metric="r2"):
    """Get test prediction from results.

    Args:
        results (bp.CompareDict): Model results.
        metric (str, optional): Metric to extract. Defaults to 'r2'.

    Returns:
        pd.DataFrame: Test predictions
    """

    if isinstance(results, str):
        results = pd.read_pickle(results)

    scores = pd.DataFrame()

    for la, m in results.items():

        lab = la.__dict__["options"]

        if len(lab) == 1:
            scope = "all"
            target = lab[0].__dict__["name"]
        else:
            scope = lab[0].__dict__["name"]
            target = lab[1].__dict__["name"]

        if scope == "cov + mri_confounds":
            continue
        else:

            best_model_idx = np.argmax(m.scores[metric])
            best_model = m.estimators[best_model_idx]

            ds = m._dataset
            X_test, y_test = ds.get_Xy(m.ps, subjects="test")
            pred = best_model.score(X_test, y_test)

            tmp = pd.DataFrame(
                {"scope": scope, "target": target, "test_r2": pred}, index=[0]
            )
            scores = pd.concat([scores, tmp], axis=0)

    return scores


def assemble_summary(
    results_path: str, params: dict, model="ridge", n_jobs=2
) -> pd.DataFrame:
    """Assemble model summary.

    Args:
        results_path (str): Path to model results.

    Returns:
        pd.DataFrame: Model summary.
    """

    results_paths = glob.glob(params["model_results_path"] + f"*{model}_results.pkl")
    summary_paths = glob.glob(params["model_results_path"] + f"*{model}_summary.csv")

    test_predictions = Parallel(n_jobs=n_jobs)(
        delayed(get_test_prediction)(r) for r in results_paths
    )
    summary = pd.concat([pd.read_csv(p) for p in summary_paths])
    summary = join_test_prediction(summary, pd.concat(test_predictions))

    summary.to_csv(results_path + f"{model}_models_summary.csv")

    return summary


def remove_nonfeatures(coefs: pd.Series, filter_strings=["mri", "iqc"]) -> pd.Series:
    return coefs[~coefs.index.str.contains("|".join(filter_strings))]


def get_feature_importance(
    results: bp.CompareDict, metric="r2"
) -> (pd.DataFrame, pd.DataFrame, pd.DataFrame):
    """Get feature importance from results.

    Args:
        results (bp.CompareDict): Model results.
        metric (str, optional): Metric to extract. Defaults to 'r2'.

    Returns:
        pd.DataFrame: Feature importance.
    """

    fis = {}
    best_fis = {}
    avg_fis = {}

    if isinstance(results, str):
        results = pd.read_pickle(results)

    for la, m in results.items():

        lab = la.__dict__["options"]

        if len(lab) == 1:
            scope = "all"
            target = lab[0].__dict__["name"]
        else:
            scope = lab[0].__dict__["name"]
            scope = scope.replace("cov + ", "")
            target = lab[1].__dict__["name"]

        if scope == "mri_confounds":
            continue
        else:
            coefs = m.get_fis()
            fis[target] = coefs

            avg_coefs = coefs.mean()
            avg_coefs = remove_nonfeatures(avg_coefs)
            avg_fis[target] = avg_coefs

            best_model_idx = np.argmax(m.scores[metric])
            best_coefs = coefs.iloc[best_model_idx, :]
            best_coefs = remove_nonfeatures(best_coefs)
            best_fis[target] = best_coefs

    return fis, best_fis, avg_fis


def assemble_feature_importance(
    fpath: str, params: dict, model: str = "ridge", n_jobs=2
) -> None:
    """Implement gather_feature_importance."""

    results_paths = glob.glob(params["model_results_path"] + "*ridge_results.pkl")
    results_paths = [path for path in results_paths if "all" not in path]
    res = Parallel(n_jobs=n_jobs)(
        delayed(get_feature_importance)(r) for r in results_paths
    )
    pd.to_pickle(res, fpath + f"{model}_feature_importance.pkl")


def main():

    params = load_yaml("../parameters.yaml")
    results_path = params["model_results_path"]

    assemble_summary(results_path, params, model="ridge")
    assemble_feature_importance(results_path, params, model="ridge")

    produce_plots(params)


if __name__ == "__main__":
    main()
