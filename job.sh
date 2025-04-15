#!/bin/bash
#SBATCH --job-name=ds2d-train
#SBATCH --output=log/job_output.log
#SBATCH --error=log/job_error.log
#SBATCH --nodes=2
#SBATCH --mem=64G 
#SBATCH --cpus-per-gpu=3
#SBATCH --gres=gpu:h100:4
#SBATCH --time=03:00:00
#SBATCH --account=aip-bengioy

# Source the .env file
# if [ -f .env ]; then
#     export $(grep -v '^#' .env | xargs)
# fi

export WANDB_MODE=offline

module load python/3.11
module load arrow
module load opencv
module load cuda/12
source $SCRATCH/env/ds2d/bin/activate

EPOCHS=${2:-30} 
OUTPUT_DIR="output/procthor_${EPOCHS}/"

torchrun --nnodes 2 --nproc_per_node 4 finetuning.py \
    --use_peft \
    --peft_method lora \
    --quantization 4bit \
    --model_name models/Llama-3.3-70B-Instruct \
    --batch_size_training 2 \
    --num_epochs $EPOCHS \
    --dataset custom_dataset \
    --context_length 4096 \
    --enable_fsdp True \
    --custom_dataset.file "floorplan_dataset.py" \
    --output_dir $OUTPUT_DIR \
    --use_wandb True \
    --wandb_config.project "floorplans"

# python -m llama_cookbook.finetuning \
#     --use_peft \
#     --peft_method lora \
#     --quantization \
#     --model_name models/Llama-3.1-8B-Instruct \
#     --batch_size_training 2 \
#     --num_epochs $EPOCHS \
#     --dataset "custom_dataset" \
#     --custom_dataset.file "floorplan_dataset.py" \
#     --custom_dataset.data_path $DATA \
#     --output_dir $OUTPUT_DIR \
#     --use_wandb True \
#     --wandb_config.project "floorplans" \
#     --train_config.max_train_step 100
