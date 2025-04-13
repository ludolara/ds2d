#!/bin/bash
#SBATCH --job-name=ds2d-train
#SBATCH --output=log/job_output.log
#SBATCH --error=log/job_error.log
#SBATCH --nodes=2
#SBATCH --mem=64G 
#SBATCH --cpus-per-gpu=2
#SBATCH --gres=gpu:h100:4
#SBATCH --time=24:00:00
#SBATCH --account=aip-bengioy

module load python/3.11
module load arrow
module load cuda/12
source $SCRATCH/env/ds2d/bin/activate

EPOCHS=${2:-10} 
OUTPUT_DIR="output/procthor_${EPOCHS}/"

nodes=( $( scontrol show hostnames $SLURM_JOB_NODELIST ) )
nodes_array=($nodes)
num_nodes=${#nodes[@]}
head_node=${nodes_array[0]}
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)
echo Node IP: $head_node_ip
echo Number of nodes: $num_nodes
# export LOGLEVEL=INFO
export WANDB_MODE=offline

srun torchrun --nnodes $num_nodes --nproc_per_node 4 --rdzv_id $RANDOM --rdzv_backend c10d --rdzv_endpoint $head_node_ip:29500 ./src/finetuning.py \
    --use_peft \
    --peft_method lora \
    --quantization 4bit \
    --model_name models/Llama-3.1-8B-Instruct \
    --batch_size_training 2 \
    --num_epochs $EPOCHS \
    --dataset custom_dataset \
    --context_length 4096 \
    --enable_fsdp True \
    --custom_dataset.file "src/floorplan_dataset.py" \
    --output_dir $OUTPUT_DIR \
    --use_wandb True \
    --wandb_config.project "floorplans"
