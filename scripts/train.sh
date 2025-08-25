#!/bin/bash
#SBATCH --job-name=ds2d-train
#SBATCH --output=log/job_output.log
#SBATCH --error=log/job_error.log
#SBATCH --nodes=1
#SBATCH --mem=64G 
#SBATCH --cpus-per-gpu=3
#SBATCH --gres=gpu:h100:4
#SBATCH --time=03:00:00

export PYTHONPATH="$PYTHONPATH:/."

module load python/3.11
module load arrow
module load cuda/12
source $SCRATCH/env/ds2d/bin/activate

EPOCHS=${2:-30} 
OUTPUT_DIR="output/v3_${EPOCHS}/"

export WANDB_MODE=offline

torchrun --nnodes 1 --nproc_per_node 4 ./src/train/finetuning.py \
    --use_peft \
    --peft_method lora \
    --quantization 4bit \
    --model_name models/Llama-3.1-8B-Instruct \
    --custom_dataset.data_path "datasets/rplan_converted_no_doors" \
    --batch_size_training 2 \
    --num_epochs $EPOCHS \
    --dataset custom_dataset \
    --context_length 4096 \
    --enable_fsdp True \
    --custom_dataset.file "src/train/floorplan_dataset.py" \
    --output_dir $OUTPUT_DIR \
    --use_wandb True \
    --wandb_config.project "floorplans"
