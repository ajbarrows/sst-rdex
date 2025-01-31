#!/bin/bash
#SBATCH --job-name=lasso_incorrect_go
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=24:00:00
#SBATCH --partition=general
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=256G
#SBATCH --mail-type=ALL

source ${HOME}/.bashrc
conda activate sst-rdex

cd ../pipelines/

python3 run_model.py lasso incorrect_go
