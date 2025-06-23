#!/bin/bash
#SBATCH --job-name=ds2d-inference
#SBATCH --output=log_inference/job_output.log
#SBATCH --error=log_inference/job_error.log
#SBATCH --nodes=1
#SBATCH --mem=64G 
#SBATCH --cpus-per-gpu=3
#SBATCH --gres=gpu:h100:4
#SBATCH --time=2:59:00
#SBATCH --account=aip-pal

export PYTHONPATH="$PYTHONPATH:/."
export VLLM_USE_V1=0

module load cuda/12
module load python/3.11
module load arrow
source $SCRATCH/env/vllm/bin/activate

TEST_RANGE=${1:-"1,768"}

python src/pred/run_generation.py \
    --batch_size 32 \
    --model_name_or_path "output/results_70B_r256_GRPO_7n/checkpoint-1100" \
    --feedback_iterations 1 \
    --dataset_name_or_path "hf_datasets/rplan" \
    --output_dir "results_70B_r256_GRPO_7n_cp1100_sampling/generations/rplan_7_70B/full_prompt" \
    --test_range "$TEST_RANGE" \

# python src/pred/run_generation.py \
#     --batch_size 32 \
#     --model_name_or_path "models/Llama-3.3-70B-Instruct" \
#     --lora_adapter_path "output/results_70B" \
#     --feedback_iterations 1 \
#     --dataset_name_or_path "hf_datasets/rplan" \
#     --output_dir "results_70B_r256_GRPO_7n_cp1100_sampling/generations/rplan_7_70B/full_prompt" \
#     --test_range "$TEST_RANGE" \
