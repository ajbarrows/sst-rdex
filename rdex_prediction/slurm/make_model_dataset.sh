#!/bin/bash
#SBATCH --job-name=make_model_dataset
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=1:00:00
#SBATCH --partition=general
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=256G
#SBATCH --mail-type=ALL


echo "Submission Dir:  ${SLURM_SUBMIT_DIR}"
echo "Running host:    ${SLURMD_NODENAME}"
echo "Assigned nodes:  ${SLURM_JOB_NODELIST}"
echo "Job ID:          ${SLURM_JOBID}"
echo "Job Name:        ${SLURM_JOB_NAME}"
echo "Partition:       ${SLURM_JOB_PARTITION}"
echo "CPUS per task:   ${SLURM_CPUS_PER_TASK}"

source ${HOME}/.bashrc
conda activate sst-rdex

cd ../pipelines/


python3 make_model_dataset.py
