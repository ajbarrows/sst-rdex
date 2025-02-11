#!/bin/bash
#SBATCH --job-name=roi_all
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=24:00:00
#SBATCH --partition=general
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
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

python3 run_model.py ridge all \
 --dataset="../../data/04_model_input/rdex_prediction_roi_dataset.pkl" \
 --scopes="../../data/04_model_input/rdex_prediction_roi_scopes.pkl" \
 --append="roi" \
 --n_cores=$SLURM_CPUS_PER_TASK

 python3 run_model.py elastic all \
 --dataset="../../data/04_model_input/rdex_prediction_roi_dataset.pkl" \
 --scopes="../../data/04_model_input/rdex_prediction_roi_scopes.pkl" \
 --append="roi" \
 --n_cores=$SLURM_CPUS_PER_TASK

python3 run_model.py lasso all \
 --dataset="../../data/04_model_input/rdex_prediction_roi_dataset.pkl" \
 --scopes="../../data/04_model_input/rdex_prediction_roi_scopes.pkl" \
 --append="roi" \
 --n_cores=$SLURM_CPUS_PER_TASK
