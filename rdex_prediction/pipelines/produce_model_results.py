import glob

import pandas as pd
import numpy as np


from joblib import Parallel, delayed

import BPt as bp
from abcd_tools.utils.ConfigLoader import load_yaml

from rdex_plotting_functions import (
    relabel_plotting_data,
    sort,
    produce_effectsize_plot,
)


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
    if "scope" in results_summary.columns:
        idx = ["target", "scope"]
    else:
        idx = "target"

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

    if "scope" not in summary.columns:
        summary["scope"] = "all"

    summary["scope"] = summary["scope"].fillna("all")

    summary = join_test_prediction(summary, pd.concat(test_predictions))

    summary.to_csv(results_path + f"{model}_models_summary.csv")

    return summary


def remove_nonfeatures(
    coefs: pd.Series, filter_strings=["serial", "motion"]
) -> pd.Series:
    return coefs[~coefs.index.str.contains("|".join(filter_strings))]


def haufe_transform(model_weights, X, batch_size=1000):
    """
    Memory-efficient Haufe transformation with corrected sign.
    Simply negate the result of the original implementation.

    Parameters:
    -----------
    model_weights : array-like, shape (n_features,)
        Weight vector from a linear model
    X : array-like, shape (n_samples, n_features)
        Feature matrix used to train the model
    batch_size : int, default=1000
        Size of batches to process at once

    Returns:
    --------
    activation_patterns : array, shape (n_features,)
        Transformed weights with correct sign
    """
    X = np.asarray(X)
    model_weights = np.asarray(model_weights).flatten()
    n_samples, n_features = X.shape

    # Compute means
    means = np.mean(X, axis=0)

    # Compute using original implementation
    activation_patterns = np.zeros(n_features)

    for i in range(0, n_samples, batch_size):
        batch = X[i : min(i + batch_size, n_samples)]
        centered_batch = batch - means
        activation_patterns += np.dot(
            centered_batch.T, np.dot(centered_batch, model_weights)
        )

    # Normalize by n_samples - 1
    activation_patterns /= n_samples - 1

    # FIX: Negate the result to correct the sign
    return -activation_patterns  # Simply negate the result


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
    haufe_avg = {}
    haufe_fis = {}

    if isinstance(results, str):
        results = pd.read_pickle(results)

    # get dataset from single model
    m = results["EEA"]
    X, _ = m._dataset.get_Xy()
    train_subjects = m.train_subjects

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

            # get Haufe-transformed features

            haufe_features = pd.Series(dtype="float64")

            drop_cols = ["mri_info_deviceserialnumber", "iqc_sst_all_mean_motion"]
            for fold_idx in range(len(coefs)):
                features = remove_nonfeatures(coefs.iloc[fold_idx])
                tmp = X[X.index.isin(train_subjects[fold_idx])]
                tmp = tmp.drop(columns=drop_cols)
                haufe = pd.Series(haufe_transform(features, tmp), index=tmp.columns)
                haufe_features = pd.concat([haufe_features, haufe], axis=1)

            haufe_mean = haufe_features.mean(axis=1)
            haufe_avg[target] = haufe_mean
            haufe_fis[target] = haufe_features

    return fis, best_fis, avg_fis, haufe_avg, haufe_fis


def assemble_feature_importance(
    fpath: str, params: dict, model: str = "ridge", n_jobs=2
) -> None:
    """Implement gather_feature_importance."""

    results_paths = glob.glob(params["model_results_path"] + f"*{model}_results.pkl")

    # results_paths = [path for path in results_paths if "all" not in path]

    if len(results_paths) > 1:
        res = Parallel(n_jobs=n_jobs)(
            delayed(get_feature_importance)(r) for r in results_paths
        )
    else:
        res = get_feature_importance(results_paths[0])

    pd.to_pickle(res, fpath + f"{model}_feature_importance.pkl")
    print(f"Feature importance saved to {fpath}")


def load_vertexwise_model_summaries(params):

    lasso = pd.read_csv(
        params["model_results_path"] + "vertex_lasso_models_summary.csv"
    )
    ridge = pd.read_csv(
        params["model_results_path"] + "vertex_ridge_models_summary.csv"
    )

    process_map = params["process_map"]
    target_map = params["target_map"]
    color_map = params["color_map"]

    lasso["scope"] = "lasso"
    ridge["scope"] = "ridge"

    return (
        pd.concat([lasso, ridge])
        .pipe(relabel_plotting_data, process_map, target_map, color_map)
        .pipe(sort)
    )


def main():

    params = load_yaml("../parameters.yaml")
    params["model_results_path"]

    # assemble_feature_importance(model_res_path, params)

    # produce_supplement_plots(params)

    produce_effectsize_plot(params, model="vertex_ridge")

    # reviewer response
    # model = "vertex_ridge_lasso"
    # summary = load_vertexwise_model_summaries(params)
    # make_effect_compare_plot(
    #     summary,
    #     model,
    #     params["effectsize_plot_title"],
    #     params["plot_output_path"] + f"{model}_effectsize_plot",
    # )


if __name__ == "__main__":
    main()
