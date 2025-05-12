import os
import argparse

import pandas as pd
import numpy as np

import BPt as bp


from rdex_predict_functions import filter_scopes, fit_model, save_model_results
from abcd_tools.utils.ConfigLoader import load_yaml


def fit(
    ds: bp.Dataset,
    condition: str,
    scopes: dict,
    model: str,
    fpath: str,
    n_cores: int,
    random_state: int,
    append: str = None,
) -> None:

    if condition == "all":
        complete = True
    else:
        scopes = filter_scopes([condition], scopes)
        complete = False

    # deal with inf values that sneak in
    ds = ds.replace([np.inf, -np.inf], np.nan).dropna()

    res = fit_model(
        ds, scopes, model, complete=complete, n_cores=n_cores, random_state=random_state
    )

    save_model_results(res, condition, model, fpath, append=append)


def main():

    params = load_yaml("../parameters.yaml")

    parser = argparse.ArgumentParser(description="Run RDEX prediction pipeline")
    parser.add_argument("model", type=str, help="Options: ridge, lasso, elastic")
    parser.add_argument(
        "condition",
        type=str,
        help="Options: all, correct_go, correct_stop, incorrect_go, incorrect_stop",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=params["sst_dataset_path"],
        help="Path to dataset",
    )
    parser.add_argument(
        "--scopes", type=str, default=params["sst_scopes_path"], help="Path to scopes"
    )
    parser.add_argument(
        "--fpath",
        type=str,
        default=params["model_results_path"],
        help="Path to save results",
    )
    parser.add_argument(
        "--n_cores",
        type=int,
        default=-1,
        help="Number of cores to use. Default: -1 (all)",
    )
    parser.add_argument("--random_state", type=int, default=42, help="Random state.")

    parser.add_argument("--append", type=str, default="", help="Append to file name")
    args = parser.parse_args()

    ds = pd.read_pickle(args.dataset)
    scopes = pd.read_pickle(args.scopes)

    cores = os.cpu_count()
    if args.n_cores == -1:
        args.n_cores = cores

    print(
        f"Fitting mod. {args.model} for cond {args.condition} with {args.n_cores} cores"
    )

    fit(
        ds,
        scopes=scopes,
        condition=args.condition,
        model=args.model,
        fpath=args.fpath,
        n_cores=args.n_cores,
        random_state=args.random_state,
        append=args.append,
    )


if __name__ == "__main__":
    main()
