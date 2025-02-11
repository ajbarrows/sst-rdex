import pandas as pd

import BPt as bp
from BPt.extensions import LinearResidualizer

from sklearn.linear_model import ElasticNet
from sklearn.preprocessing import OneHotEncoder


def filter_scopes(keys: list, scopes: dict) -> dict:
    keys.extend(["mri_confounds"])
    return {k: v for k, v in scopes.items() if k in keys}


def define_regression_pipeline(ds: bp.Dataset, scopes: dict, model: str) -> bp.Pipeline:

    # Just scale float type features
    robust_scaler = bp.Scaler("robust", scope="float")

    # Define residualization procedure
    ohe = OneHotEncoder(
        categories="auto", drop="if_binary", sparse=False, handle_unknown="ignore"
    )
    ohe_tr = bp.Transformer(ohe, scope="category")

    resid = LinearResidualizer(to_resid_df=ds["covariates"], fit_intercept=True)
    resid_tr = bp.Scaler(resid, scope=list(scopes.keys()))

    # Define regression model
    mod_params = {"alpha": bp.p.Log(lower=1e-5, upper=1e5)}

    if model == "ridge":
        mod_obj = "ridge"
    elif model == "elastic":
        mod_obj = ElasticNet()
        l1_ratio = bp.p.Scalar(lower=0.001, upper=1).set_mutation(sigma=0.165)
        mod_params["l1_ratio"] = l1_ratio
    elif model == "lasso":
        mod_obj = "lasso regressor"

    param_search = bp.ParamSearch("HammersleySearch", n_iter=100, cv="default")
    model = bp.Model(obj=mod_obj, params=mod_params, param_search=param_search)

    return bp.Pipeline([robust_scaler, ohe_tr, resid_tr, model])


def fit_model(
    ds: bp.Dataset,
    scopes: dict,
    model="ridge",
    complete=False,
    n_cores=1,
    random_state=42,
) -> bp.CompareDict:

    if model in ["elastic", "ridge", "lasso"]:
        pipe = define_regression_pipeline(ds, scopes, model=model)
    else:
        raise Exception(f"Specified model {model} is not implemented")

    if not complete:
        compare_scopes = []
        for key in scopes.keys():
            if key != "mri_confounds":
                compare_scopes.append(
                    bp.Option(["covariates", key], name=f"cov + {key}")
                )
        compare_scopes = bp.Compare(compare_scopes)
    else:
        compare_scopes = None

    ps = bp.ProblemSpec(n_jobs=n_cores, random_state=random_state)
    cv = bp.CV(splits=5, n_repeats=1)

    results = bp.evaluate(
        pipeline=pipe,
        dataset=ds,
        problem_spec=ps,
        scope=compare_scopes,
        mute_warnings=True,
        verbose=0,
        target=bp.Compare(ds.get_cols("target")),
        cv=cv,
    )

    return results


def save_model_results(
    res: bp.CompareDict, name: str, model: str, path: str, append: bool
) -> None:
    """Save model results to disk.

    Args:
        res (bp.CompareDict): Model results.
        name (str): Model name.
        model (str): Model type.
        path (str): Path to save results.
    """

    if append:
        name = f"{name}_{append}"

    pd.to_pickle(res, f"{path}/{name}_{model}_results.pkl")

    summary = res.summary()
    summary.to_csv(f"{path}/{name}_{model}_summary.csv")

    print(f"Results saved to {path}")
