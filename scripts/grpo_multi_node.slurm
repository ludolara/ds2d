#!/bin/bash
#SBATCH --job-name=train-grpo-70B-5
#SBATCH --output=logs/train_grpo_70B/job_output.log
#SBATCH --error=logs/train_grpo_70B/job_error.log
#SBATCH --nodes=5
#SBATCH --mem=256G
#SBATCH --cpus-per-gpu=3
#SBATCH --gres=gpu:h100:4
#SBATCH --time=5:59:00
#SBATCH --account=aip-pal

export LOGLEVEL=INFO
export WANDB_MODE=offline

module load python/3.11
module load opencv
module load arrow
module load cuda/12
source $SCRATCH/env/vllm/bin/activate

ROOM_NUMBER=${1:-5}

MODEL="models/final/rplan${ROOM_NUMBER}_r64_a128-Llama-3.3-70B-Instruct"
DATASET="datasets/final/rplan_${ROOM_NUMBER}"
OUTPUT_DIR="/home/l/luislara/links/projects/aip-pal/luislara/output/rplan${ROOM_NUMBER}_70B_r64_GRPO_9n"

NODELIST=($(scontrol show hostnames $SLURM_JOB_NODELIST))
NUM_NODES=${#NODELIST[@]}
NUM_TRAIN_NODES=$((NUM_NODES - 1))
TRAIN_NODES=${NODELIST[@]:0:$NUM_TRAIN_NODES}
VLLM_NODE=${NODELIST[$NUM_TRAIN_NODES]}
NUM_PROCESSES=$((NUM_TRAIN_NODES * 4))
HEAD_NODE=${NODELIST[0]}
HEAD_NODE_IP=$(srun --nodes=1 --ntasks=1 -w "$HEAD_NODE" hostname --ip-address)

echo "Training nodes:        ${TRAIN_NODES[@]}"
echo "vLLM node:             $VLLM_NODE"
echo "Total processes:       $NUM_PROCESSES"
echo "Head node IP:          $HEAD_NODE_IP"

# Run training on the first n-1 nodes
srun --nodes=$NUM_TRAIN_NODES --ntasks=$NUM_TRAIN_NODES --nodelist="${TRAIN_NODES}" accelerate launch \
     --config_file src/grpo/accelerate_configs/deepspeed_zero3.yaml \
     --num_processes $NUM_PROCESSES \
     --num_machines $NUM_TRAIN_NODES \
     --main_process_ip $HEAD_NODE_IP \
     --main_process_port 29500 \
     --machine_rank $SLURM_PROCID  \
     --rdzv_backend c10d \
     src/grpo/train_grpo.py \
     --output "${OUTPUT_DIR}" \
     --model "${MODEL}" \
     --dataset "${DATASET}" \
     --vllm_server_host $VLLM_NODE &

# Run vLLM server on the last node 
srun --nodes=1 --ntasks=1 --nodelist="${VLLM_NODE}" trl vllm-serve --model "${MODEL}" --tensor_parallel_size 4 --log_level warning --dtype bfloat16 &

wait

