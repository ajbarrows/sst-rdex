#!/bin/bash
#SBATCH --job-name=rdex_res
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=02:00:00
#SBATCH --partition=general
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --mail-type=ALL
#SBATCH --mail-user=ajbarrow@uvm.edu

echo "Submission Dir:  ${SLURM_SUBMIT_DIR}"
echo "Running host:    ${SLURMD_NODENAME}"
echo "Assigned nodes:  ${SLURM_JOB_NODELIST}"
echo "Job ID:          ${SLURM_JOBID}"
echo "Job Name:        ${SLURM_JOB_NAME}"
echo "Partition:       ${SLURM_JOB_PARTITION}"
echo "CPUS per task:   ${SLURM_CPUS_PER_TASK}"

set -x
source ${HOME}/.bashrc
conda activate sst-rdex

cd ../pipelines/

python3 produce_model_results.py
