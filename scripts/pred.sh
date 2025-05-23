#!/bin/bash
#SBATCH --job-name=ds2d-inference
#SBATCH --output=log_inference/job_output.log
#SBATCH --error=log_inference/job_error.log
#SBATCH --nodes=1
#SBATCH --mem=64G 
#SBATCH --cpus-per-gpu=3
#SBATCH --gres=gpu:h100:4
#SBATCH --time=3:00:00
#SBATCH --account=aip-pal

export PYTHONPATH="$PYTHONPATH:/."
export VLLM_USE_V1=0

module load python/3.11
module load arrow
source $SCRATCH/env/vllm/bin/activate

TEST_RANGE=${1:-"1,768"}

python src/pred/run_generation.py \
    --batch_size 8 \
    --model_name_or_path "output/8B-GRPO_3/checkpoint-150" \
    --feedback_iterations 1 \
    --dataset_name_or_path "datasets/rplan_converted_no_doors" \
    --output_dir "results/generations/rplan_25_70B/full_prompt" \
    --test_range "$TEST_RANGE" \

# python src/pred/run_generation.py \
#     --batch_size 8 \
#     --model_name_or_path "models/Llama-3.1-8B-Instruct" \
#     --lora_adapter_path "output/rplan_30_8B_no_doors_r_32" \
#     --feedback_iterations 1 \
#     --dataset_name_or_path "datasets/rplan_converted_no_doors" \
#     --output_dir "results/generations/rplan_25_70B/full_prompt" \
#     --test_range "$TEST_RANGE" \
