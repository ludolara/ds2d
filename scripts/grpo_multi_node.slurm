#!/bin/bash
#SBATCH --job-name=ds2d-grpo
#SBATCH --output=logs/train_grpo/job_output.log
#SBATCH --error=logs/train_grpo/job_error.log
#SBATCH --nodes=3
#SBATCH --mem=64G
#SBATCH --cpus-per-gpu=3
#SBATCH --gres=gpu:h100:4
#SBATCH --time=24:00:00
#SBATCH --account=aip-pal

module load python/3.11
module load opencv
module load arrow
source $SCRATCH/env/vllm/bin/activate
export LOGLEVEL=INFO
export WANDB_MODE=offline

NODELIST=($(scontrol show hostnames $SLURM_JOB_NODELIST))

# Assign the first 2 nodes for training and the 5th node for vLLM
TRAIN_NODES="${NODELIST[@]:0:2}"  # Nodes 0, 1 for training
VLLM_NODE="${NODELIST[3]}"  # Node 3 for vLLM

# Run training on the first 2 nodes (Group 1)
srun --nodes=4 --ntasks=4 --nodelist="${NODELIST[@]:0:2}" accelerate launch \
     --config_file src/grpo/accelerate_configs/deepspeed_zero3.yaml \
     --num_processes 32 \
     --num_machines 1 \
     --main_process_ip ${NODELIST[0]} \
     --machine_rank $SLURM_PROCID \
     --rdzv_backend c10d \
     ../src/grpo/train_grpo.py \
     --server_ip $VLLM_NODE &

# Run vLLM server on the 3th node (Group 2)
srun --nodes=1 --ntasks=1 --nodelist="${NODELIST[4]}" trl vllm-serve --model models/Llama-3.3-70B-Instruct --tensor_parallel_size 4 &

wait
