#!/bin/bash
#SBATCH --job-name=ds2d-train-random
#SBATCH --output=logs/train_70B/job_output.log
#SBATCH --error=logs/train_70B/job_error.log
#SBATCH --nodes=6
#SBATCH --mem=192G
#SBATCH --cpus-per-gpu=3
#SBATCH --gres=gpu:h100:4
#SBATCH --time=23:59:00
#SBATCH --account=aip-pal

export PYTHONPATH="$PYTHONPATH:/."

module load python/3.11
module load arrow
module load cuda/12
source $SCRATCH/env/ds2d/bin/activate

EPOCHS=${1:-30} 
OUTPUT_DIR="output/rplan_${EPOCHS}_70B_no_doors_r_128/"

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
    --use_peft True \
    --custom_dataset.data_path "hf_datasets/rplan_converted_no_doors" \
    --custom_dataset.file "src/train/floorplan_dataset.py" \
    --batch_size_training 2 \
    --num_epochs $EPOCHS \
    --dataset custom_dataset \
    --context_length 4096 \
    --enable_fsdp True \
    --num_workers_dataloader 32 \
    --output_dir $OUTPUT_DIR \
    --use_wandb True \
    --wandb_config.project "floorplans"
