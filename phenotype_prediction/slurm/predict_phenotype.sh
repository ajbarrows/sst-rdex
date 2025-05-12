#!/bin/bash
#SBATCH --job-name=rdex_phenotype
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=1:00:00
#SBATCH --partition=general
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --mail-type=ALL

source ${HOME}/.bashrc
conda activate sst-rdex

cd ../pipelines/

python predict_phenotype.py
