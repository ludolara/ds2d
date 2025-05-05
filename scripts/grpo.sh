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
module load opencv
module load arrow
module load cuda/12
source $SCRATCH/env/vllm/bin/activate

export LOGLEVEL=INFO
export WANDB_MODE=offline

# accelerate launch --num_processes 4 src/grpo/example_train_grpo.py 
accelerate launch --config_file src/grpo/accelerate_configs/deepspeed_zero3.yaml src/grpo/train_grpo.py 
