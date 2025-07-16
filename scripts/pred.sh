#!/bin/bash
#SBATCH --job-name=ds2d-inference
#SBATCH --output=log_inference/job_output.log
#SBATCH --error=log_inference/job_error.log
#SBATCH --nodes=1
#SBATCH --mem=64G 
#SBATCH --cpus-per-gpu=3
#SBATCH --gres=gpu:h100:4
#SBATCH --time=11:59:00
#SBATCH --account=aip-pal

export PYTHONPATH="$PYTHONPATH:/."

module load cuda/12
module load python/3.11
module load arrow
source $SCRATCH/env/vllm/bin/activate

TEST_RANGE=${1:-"1,8500"}

python src/pred/run_generation.py \
    --batch_size 64 \
    --model_name_or_path "/home/l/luislara/links/projects/aip-pal/luislara/output/70B_r64_GRPO_9n/checkpoint-400" \
    --dataset_name_or_path "datasets/rplan_8" \
    --output_dir "results_GRPO_70B_ckpt400_all_sampling/generations/rplan_8_70B/full_prompt" \
    --test_range "$TEST_RANGE" \
    --use_sampling 

# --model_name_or_path "/home/l/luislara/links/projects/aip-pal/luislara/output/70B_r256_GRPO_9n/checkpoint-1600" \


# python src/pred/run_generation.py \
#     --batch_size 64 \
#     --model_name_or_path "models/Llama-3.3-70B-Instruct" \
#     --lora_adapter_path "output/rplan_3_70B_r64_a128_all" \
#     --dataset_name_or_path "datasets/rplan_8" \
#     --output_dir "results_70B_r64_a128_all/generations/rplan_8_70B/full_prompt" \
#     --test_range "$TEST_RANGE" 
