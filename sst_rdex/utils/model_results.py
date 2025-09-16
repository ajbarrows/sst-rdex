"""Shared utilities for processing model results across modules."""

import pandas as pd
import BPt as bp


def get_full_summary(res: bp.CompareDict) -> pd.DataFrame:
    """Helper to get full summary information from BPt models.

    Args:
        res (bp.CompareDict): Model results.

    Returns:
        pd.DataFrame: Full summary information
    """

    keys = list(res.keys())
    repr_key = keys[0]
    option_keys = [o.key for o in repr_key.options]
    cols = {key: [] for key in option_keys}

    for key in keys:
        for option in key.options:
            cols[option.key].append(option.name)

        evaluator = res[key]

        attr = getattr(evaluator, "scores")
        new_col_names = []
        for key in attr:

            val = attr[key]

            new_col_name = "scores" + "_" + key
            new_col_names.append(new_col_name)

            try:
                cols[new_col_name].append(val)
            except KeyError:
                cols[new_col_name] = [val]

    s = pd.DataFrame.from_dict(cols)
    return s.explode(new_col_names)


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
    import numpy as np

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
