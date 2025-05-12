#!/bin/bash
#SBATCH --job-name=ridge_notf
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=48:00:00
#SBATCH --partition=week
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=512G
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
 --dataset="../../data/04_model_input/rdex_prediction_dataset_no_tf.pkl" \
 --append="vertex_no_tf" \
 --n_cores=$SLURM_CPUS_PER_TASK
