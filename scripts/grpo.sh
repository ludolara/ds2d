#!/bin/bash

module load python/3.11
module load opencv
module load arrow
module load cuda/12
source $SCRATCH/env/vllm/bin/activate

export LOGLEVEL=INFO
export WANDB_MODE=offline

accelerate launch --config_file src/grpo/accelerate_configs/deepspeed_zero3.yaml src/grpo/train_grpo_1.py 
