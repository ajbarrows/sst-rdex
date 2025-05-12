import pandas as pd

import argparse


from abcd_tools.utils.ConfigLoader import load_yaml


def load_results(fpath, key: str):
    return pd.read_pickle(fpath)[key]


def run_permutation(res, n_permutations=1):
    return res.run_permutation_test(n_perm=n_permutations)


def save_results(res, fpath):
    pd.to_pickle(res, fpath)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Run permutation tests on model results."
    )
    parser.add_argument("key", type=str, help="bp.Options to identify model.")
    parser.add_argument(
        "--n_permutations",
        type=int,
        default=100,
        help="Number of permutations to run.",
    )

    args = parser.parse_args()

    params = load_yaml("../parameters.yaml")
    results = load_results(
        params["model_results_path"] + "all_vertex_ridge_results.pkl", key=args.key
    )

    # Run permutation test
    res = run_permutation(results, n_permutations=args.n_permutations)

    # Save results
    save_results(
        res,
        params["model_results_path"] + f"permutation/permutation_test_{args.key}.pkl",
    )
