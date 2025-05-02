#!/bin/bash
#SBATCH --job-name=ds2d-train-grpo
#SBATCH --output=logs/train_grpo/job_output.log
#SBATCH --error=logs/train_grpo/job_error.log
#SBATCH --nodes=1
#SBATCH --mem=48G
#SBATCH --cpus-per-gpu=3
#SBATCH --gres=gpu:h100:4
#SBATCH --time=12:00:00
#SBATCH --account=aip-pal

module load python/3.11
module load arrow
source $SCRATCH/env/vllm/bin/activate

accelerate launch src/grpo/test_train_grpo.py