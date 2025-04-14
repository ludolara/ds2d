#!/bin/bash
#SBATCH --job-name=ds2d-inference
#SBATCH --output=log_inference/job_output.log
#SBATCH --error=log_inference/job_error.log
#SBATCH --nodes=1
#SBATCH --mem=64G 
#SBATCH --cpus-per-gpu=3
#SBATCH --gres=gpu:h100:4
#SBATCH --time=06:00:00
#SBATCH --account=aip-pal

export PYTHONPATH="$PYTHONPATH:/."

module load python/3.11
module load arrow
module load cuda/12
source $SCRATCH/env/ds2d/bin/activate

TEST_RANGE=${1:-"1,700"}

python src/pred/run_generation.py \
    --with_feedback \
    --test_range "$TEST_RANGE" \
    --batch_size 4
