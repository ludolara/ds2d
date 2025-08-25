#!/bin/bash
#SBATCH --job-name=ds2d-inference
#SBATCH --output=log_inference/job_output.log
#SBATCH --error=log_inference/job_error.log
#SBATCH --nodes=1
#SBATCH --mem=64G 
#SBATCH --cpus-per-gpu=1
#SBATCH --gres=gpu:h100:4
#SBATCH --time=2:59:00
#SBATCH --account=aip-pal

export PYTHONPATH="$PYTHONPATH:/."

module load cuda/12
module load python/3.11
module load arrow

source $SCRATCH/env/vllm/bin/activate

TEST_RANGE=${1:-"1,1000"}
ROOM_NUMBER=${2:-6}

python src/pred/run_generation.py \
    --batch_size 64 \
    --model_name_or_path "models/Llama-3.3-70B-Instruct" \
    --dataset_name_or_path "datasets/final/rplan_${ROOM_NUMBER}" \
    --output_dir "results/results${ROOM_NUMBER}_GRPO_70B_fs" \
    --test_range "$TEST_RANGE" \
    --use_sampling 

# --model_name_or_path "/home/l/luislara/links/projects/aip-pal/luislara/output/rplan${ROOM_NUMBER}_70B_r64_GRPO_7n/checkpoint-100" \

# python src/pred/run_generation.py \
#     --batch_size 64 \
#     --model_name_or_path "models/Llama-3.3-70B-Instruct" \
#     --lora_adapter_path "output/final_2/rplan${ROOM_NUMBER}_2_70B_r64_a128_all" \
#     --dataset_name_or_path "datasets/final_2/rplan_${ROOM_NUMBER}" \
#     --output_dir "results/results${ROOM_NUMBER}_70B_2/" \
#     --test_range "$TEST_RANGE" \
#     --use_sampling 
