import argparse
import subprocess
import time

from abcd_tools.utils.ConfigLoader import load_yaml


def generate_slurm_script(
    model: str,
    n_permutations: int,
    walltime: str = "120:00:00",
    mem: str = "64G",
    envname: str = "sst-rdex",
    fpath: str = "../slurm/",
    partition: str = "week",
    launch=False,
) -> str:
    script = f"""#!/bin/bash
#SBATCH --job-name={model}_permutations
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time={walltime}
#SBATCH --partition={partition}
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem={mem}
#SBATCH --mail-type=ALL

source ${{HOME}}/.bashrc
conda activate {envname}

cd ../pipelines/

python3 permute_models.py {model} --n_permutations {n_permutations}
"""
    with open(f"{fpath}/{model}_permutations.sh", "w") as f:
        f.write(script)

    print(f"Slurm script saved to {fpath}/{model}_permutations.sh")

    if launch:
        time.sleep(1)
        subprocess.run(f"cd {fpath}", shell=True)
        subprocess.run(f"sbatch {model}_permutations.sh", shell=True)

    return script


def generate_scripts(
    models: list, n_permutations: int, fpath: str, launch: bool
) -> None:
    for model in models:
        generate_slurm_script(model, n_permutations, fpath=fpath, launch=launch)


def main():

    params = load_yaml("../parameters.yaml")

    parser = argparse.ArgumentParser(
        description="Generate slurm scripts for RDEX prediction pipeline"
    )
    parser.add_argument(
        "n_permutations",
        type=int,
        help="Number of permutations to run.",
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

    models = params["process_map"].keys()

    generate_scripts(
        models, args.n_permutations, fpath=args.directory, launch=args.launch
    )


if __name__ == "__main__":
    main()
