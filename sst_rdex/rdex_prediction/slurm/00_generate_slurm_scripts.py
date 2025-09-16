import argparse
import subprocess
import time

from abcd_tools.utils.ConfigLoader import load_yaml


def generate_slurm_script(
    model: str,
    condition: str,
    walltime: str = "24:00:00",
    mem: str = "256G",
    envname: str = "sst-rdex",
    fpath: str = "../slurm/",
    launch=False,
) -> str:
    script = f"""#!/bin/bash
#SBATCH --job-name={model}_{condition}
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time={walltime}
#SBATCH --partition=general
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem={mem}
#SBATCH --mail-type=ALL

source ${{HOME}}/.bashrc
conda activate {envname}

cd ../pipelines/

python3 run_model.py {model} {condition}
"""
    with open(f"{fpath}/{model}_{condition}.sh", "w") as f:
        f.write(script)

    print(f"Slurm script saved to {fpath}/{model}_{condition}.sh")

    if launch:
        time.sleep(1)
        subprocess.run(f"cd {fpath}", shell=True)
        subprocess.run(f"sbatch {model}_{condition}.sh", shell=True)

    return script


def generate_scripts(
    model_comparisons: list, predictor_comparison: list, fpath: str, launch: bool
) -> None:
    for model in model_comparisons:
        for condition in predictor_comparison:
            generate_slurm_script(model, condition, fpath=fpath, launch=launch)


def main():

    params = load_yaml("../parameters.yaml")

    parser = argparse.ArgumentParser(
        description="Generate slurm scripts for RDEX prediction pipeline"
    )
    parser.add_argument(
        "--models",
        type=list,
        default=params["model_comparisons"],
        help="Options: ridge, lasso, elastic",
    )
    parser.add_argument(
        "--conditions",
        type=list,
        default=params["predictor_comparisons"],
        help="Options: all, correct_go, correct_stop, incorrect_go, incorrect_stop",
    )
    parser.add_argument(
        "--launch",
        type=bool,
        default=False,
        action=argparse.BooleanOptionalAction,
        help="Launch slurm scripts",
    )
    parser.add_argument(
        "--directory",
        type=str,
        default=".",
        help="Directory to save slurm scripts. Default: .",
    )

    args = parser.parse_args()

    if isinstance(args.models, str):
        args.model = [args.models]
    if isinstance(args.conditions, str):
        args.condition = [args.conditions]

    generate_scripts(
        args.models, args.conditions, fpath=args.directory, launch=args.launch
    )


if __name__ == "__main__":
    main()
