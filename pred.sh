#!/bin/bash
#SBATCH --job-name=ds2d-inference
#SBATCH --output=log_inference/job_output.log
#SBATCH --error=log_inference/job_error.log
#SBATCH --nodes=1
#SBATCH --mem=64G 
#SBATCH --cpus-per-gpu=3
#SBATCH --gres=gpu:h100:4
#SBATCH --time=3:00:00
#SBATCH --account=aip-bengioy

export PYTHONPATH="$PYTHONPATH:/."

module load python/3.11
module load arrow
source $SCRATCH/env/vllm/bin/activate

TEST_RANGE=${1:-"1,768"}

python src/pred/run_generation.py \
    --batch_size 8 \
    --model_name_or_path "models/Llama-3.3-70B-Instruct" \
    --lora_adapter_path "output/no_doors_rplan_20_70B" \
    --feedback_iterations 5 \
    --dataset_name_or_path "datasets/rplan_converted_no_doors" \
    --output_dir "results/generations/no_doors_rplan_20_70B/full_prompt" \
    --test_range "$TEST_RANGE" \
