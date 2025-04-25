#!/bin/bash
#SBATCH --job-name=ds2d-train_no_doors
#SBATCH --output=logs/train_no_doors/job_output.log
#SBATCH --error=logs/train_no_doors/job_error.log
#SBATCH --nodes=4
#SBATCH --mem=64G 
#SBATCH --cpus-per-gpu=3
#SBATCH --gres=gpu:h100:4
#SBATCH --time=12:00:00
#SBATCH --account=aip-pal

module load python/3.11
module load arrow
module load cuda/12
source $SCRATCH/env/ds2d/bin/activate

EPOCHS=${1:-20} 
OUTPUT_DIR="output/no_doors_rplan_${EPOCHS}_70B/"

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
    --custom_dataset.data_path "datasets/rplan_converted_no_doors" \
    --custom_dataset.file "src/train/floorplan_dataset_no_doors.py" \
    --batch_size_training 2 \
    --num_epochs $EPOCHS \
    --dataset custom_dataset \
    --context_length 6144 \
    --enable_fsdp True \
    --output_dir $OUTPUT_DIR \
    --use_wandb True \
    --wandb_config.project "floorplans"
