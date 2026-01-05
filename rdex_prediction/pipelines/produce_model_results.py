import glob

import pandas as pd
import numpy as np


from joblib import Parallel, delayed

import BPt as bp
from abcd_tools.utils.ConfigLoader import load_yaml

from rdex_plotting_functions import relabel_plotting_data, sort, produce_plots


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


def haufe_transform(X, y_pred, chunk_size=10000):
    """Ultra-fast chunked version for huge feature matrices"""
    n = len(X)
    X_mean = X.mean(axis=0)
    y_mean = y_pred.mean()
    y_centered = y_pred - y_mean

    if chunk_size >= n:
        X_centered = X - X_mean
        return (X_centered.T @ y_centered) / len(X)
    else:
        result = np.zeros(X.shape[1])
        for i in range(0, X.shape[1], chunk_size):
            end = min(i + chunk_size, X.shape[1])
            X_chunk = X[:, i:end] - X_mean[i:end]
            result[i:end] = X_chunk.T @ y_centered

        return result / n


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

    if isinstance(results, str):
        results = pd.read_pickle(results)

    def get_training_set(model):
        X, _ = model._dataset.get_Xy()
        train_subjects = model.train_subjects

        # return X.loc[train_subjects]
        return [X.loc[split] for split in train_subjects]
        # return train_subjects

    # get dataset from single model
    X_train_splits = get_training_set(results["EEA"])

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

            haufe_fis = []
            for i, split in enumerate(X_train_splits):
                est = m.estimators[i]
                y_pred = est.predict(split)

                haufe_fis.append(
                    haufe_transform(split, y_pred).pipe(remove_nonfeatures)
                )

            haufe_avg[target] = pd.concat(haufe_fis, axis=1).mean(axis=1)

    return fis, best_fis, avg_fis, haufe_avg


def assemble_feature_importance(
    fpath: str, params: dict, model: str = "ridge", n_jobs=2
) -> None:
    """Implement gather_feature_importance."""

    results_paths = glob.glob(
        params["model_results_path"] + f"all_vertex_{model}_results.pkl"
    )

    res = get_feature_importance(results_paths[0])

    pd.to_pickle(res, fpath + f"{model}_feature_importance.pkl")
    print(f"Feature importance saved to {fpath} + {model}_feature_importance.pkl")


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

    # assemble_summary(model_res_path, params, model="contrasts_ridge")

    # for model in ["lasso", "ridge", "contrasts_ridge"]:
    for model in ["contrasts_ridge"]:

        # assemble_feature_importance(model_res_path, params, model=model)
        produce_plots(params, model=model)

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
