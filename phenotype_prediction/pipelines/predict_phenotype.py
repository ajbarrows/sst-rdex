import pandas as pd
import os
import BPt as bp

from abcd_tools.utils.io import load_tabular
from abcd_tools.utils.ConfigLoader import load_yaml

from sklearn.linear_model import ElasticNet


def get_phenotype_scopes(predictors: pd.DataFrame, fpath: str, params: dict) -> dict:
    """Get scopes for phenotype prediction pipeline.

    Args:
        predictors (pd.DataFrame): Predictors dataset.
        fpath (str): Path to save scopes.
        params (dict): Parameters dictionary.

    Returns:
        dict: Scopes dictionary.
    """

    empirical_predictors = ["correct_go_mrt", "correct_go_stdrt", "issrt"]
    covariates = params["covariates"]
    category = params["categorical"]
    predictors = predictors.loc[:, ~predictors.columns.isin(covariates)]

    rdex_predictors = predictors.loc[
        :, ~predictors.columns.isin(empirical_predictors)
    ].columns.tolist()

    scopes = {
        "category": category,
        "covariates": covariates,
        "empirical": empirical_predictors,
        "rdex": rdex_predictors,
        "rdex + empirical": predictors.columns.tolist(),
    }

    if fpath:
        pd.to_pickle(scopes, fpath + "phenotype_prediction_scopes.pkl")

    return scopes


def prepare_phenotype_dataset(
    rdex_ds: bp.Dataset,
    predictors: pd.DataFrame,
    phenotype: pd.DataFrame,
    params: dict,
    fpath: str,
) -> bp.Dataset:
    """Prepare phenotype prediction dataset.

    Args:
        rdex_ds (bp.Dataset): RDEX dataset.
        predictors (pd.DataFrame): Predictors dataset.
        phenotype (pd.DataFrame): Phenotype dataset.
        params (dict): Parameters dictionary.
        fpath (str): Path to save dataset.

    Returns:
        bp.Dataset: Phenotype prediction dataset.
    """

    # gather index from rdex prediction dataset
    rdex_ds_train = rdex_ds.train_subjects
    rdex_ds_test = rdex_ds.test_subjects
    all_subjects = rdex_ds_train.append(rdex_ds_test)

    # exclude bpm columns from phenotype -- too much missingness
    phenotype = phenotype.loc[:, ~phenotype.columns.str.startswith("bpm")]
    df = predictors.join(phenotype, how="inner")

    # limit to subjects in rdex dataset
    df = df.loc[df.index.isin(all_subjects)]

    scopes = get_phenotype_scopes(predictors, fpath, params)
    targets = phenotype.loc[:, ~phenotype.columns.isin(params["covariates"])].columns

    ds = bp.Dataset(df, targets=targets)
    ds = ds.set_train_split(subjects=rdex_ds_train)

    for k, v in scopes.items():
        ds.add_scope(v, k, inplace=True)

    ds = ds.ordinalize(scope="category")
    ds = ds.dropna()

    if fpath:
        ds.to_pickle(fpath + "phenotype_prediction_dataset.pkl")

    return ds, scopes


def define_phenotype_prediction_pipeline() -> bp.Pipeline:
    """Define phenotype prediction pipeline.

    Returns:
        bp.Pipeline: Phenotype prediction pipeline.
    """

    # Just scale float type features
    scaler = bp.Scaler("robust", scope="float")
    normalizer = bp.Scaler("normalize", scope="float")

    # Define regression model
    mod_obj = ElasticNet()
    mod_params = {
        "alpha": bp.p.Log(lower=1e-5, upper=1e5),
        "l1_ratio": bp.p.Scalar(lower=0.001, upper=1).set_mutation(sigma=0.165),
    }
    param_search = bp.ParamSearch("HammersleySearch", n_iter=100, cv="default")

    model = bp.Model(obj=mod_obj, params=mod_params, param_search=param_search)

    # Then define full pipeline
    pipe = bp.Pipeline([scaler, normalizer, model])

    return pipe


def fit_phenotype_prediction_model(
    ds: bp.Dataset, scopes: dict, n_cores=1, random_state=42
) -> bp.CompareDict:
    """Fit phenotype prediction model.

    Args:
        ds (bp.Dataset): Phenotype prediction dataset.
        scopes (dict): Scopes dictionary.
        n_cores (int): Number of cores to use.
        random_state (int): Random state.

    Returns:
        bp.CompareDict: Model results.
    """

    pipe = define_phenotype_prediction_pipeline()
    cv = bp.CV(splits=5, n_repeats=1)
    ps = bp.ProblemSpec(n_jobs=n_cores, random_state=random_state)

    compare_scopes = []
    for key in scopes.keys():
        if key not in ["category", "covariates"]:
            compare_scopes.append(bp.Option(["covariates", key], name=key))

    results = bp.evaluate(
        pipeline=pipe,
        dataset=ds,
        problem_spec=ps,
        scope=bp.Compare(compare_scopes),
        target=bp.Compare(ds.get_cols("target")),
        mute_warnings=True,
        verbose=0,
        cv=cv,
    )

    return results


def save_model_results(res: bp.CompareDict, name: str, model: str, path: str) -> None:
    """Save model results to disk.

    Args:
        res (bp.CompareDict): Model results.
        name (str): Model name.
        model (str): Model type.
        path (str): Path to save results.
    """
    pd.to_pickle(res, f"{path}/{name}_{model}_results.pkl")

    summary = res.summary()
    summary.to_csv(f"{path}/{name}_{model}_summary.csv")

    print(f"Results saved to {path}")


def main():
    params = load_yaml("../parameters.yaml")

    predictors = load_tabular(params["behavioral_path"])
    phenotype = load_tabular(params["phenotype_path"])
    rdex_ds = pd.read_pickle(params["rdex_prediction_dataset_path"])

    ds, scopes = prepare_phenotype_dataset(
        rdex_ds, predictors, phenotype, params, params["phenotype_input_dir"]
    )

    n_cores = os.cpu_count()
    results = fit_phenotype_prediction_model(
        ds, scopes, n_cores=n_cores, random_state=42
    )
    save_model_results(results, "phenotype", "elastic", params["phenotype_output_dir"])


if __name__ == "__main__":
    main()
