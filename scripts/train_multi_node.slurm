#!/bin/bash
#SBATCH --job-name=peft-5
#SBATCH --output=logs/train_70B/job_output.log
#SBATCH --error=logs/train_70B/job_error.log
#SBATCH --nodes=6
#SBATCH --mem=196G
#SBATCH --cpus-per-gpu=2
#SBATCH --gres=gpu:h100:4
#SBATCH --time=10:59:00
#SBATCH --account=aip-pal

export PYTHONPATH="$PYTHONPATH:/."

module load python/3.11
module load arrow
module load cuda/12
source $SCRATCH/env/ds2d/bin/activate

EPOCHS=${1:-2}
ROOM_NUMBER=${2:-5}
OUTPUT_DIR="output/final_2/rplan${ROOM_NUMBER}_${EPOCHS}_70B_r64_a128_all/"

nodes=( $( scontrol show hostnames $SLURM_JOB_NODELIST ) )
nodes_array=($nodes)
num_nodes=${#nodes[@]}
head_node=${nodes_array[0]}
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)
echo Node IP: $head_node_ip
echo Number of nodes: $num_nodes
export LOGLEVEL=INFO
export WANDB_MODE=offline

srun torchrun --nnodes $num_nodes --nproc_per_node 4 --rdzv_id $RANDOM --rdzv_backend c10d --rdzv_endpoint $head_node_ip:29500 ./src/train/finetuning.py \
    --use_peft \
    --peft_method lora \
    --quantization 4bit \
    --model_name models/Llama-3.3-70B-Instruct \
    --custom_dataset.data_path "datasets/final_2/rplan_${ROOM_NUMBER}" \
    --custom_dataset.file "src/train/floorplan_dataset.py" \
    --batch_size_training 2 \
    --lora_config.r 64 \
    --lora_config.lora_alpha 128 \
    --num_epochs $EPOCHS \
    --dataset custom_dataset \
    --context_length 6144 \
    --enable_fsdp True \
    --num_workers_dataloader 32 \
    --output_dir $OUTPUT_DIR \
    --use_wandb True \
    --wandb_config.project "ds2d"

# --from_peft_checkpoint "output/rplan_25_70B" \
