#!/bin/bash
#SBATCH --job-name=ds2d-inference
#SBATCH --output=log/job_output.log
#SBATCH --error=log/job_error.log
#SBATCH --ntasks=1
#SBATCH --partition=long
#SBATCH --gres=gpu:l40s:2
#SBATCH --cpus-per-task=4
#SBATCH --mem=32GB
#SBATCH --time=24:00:00

export PYTHONPATH="$PYTHONPATH:/."

module load python/3.11
module load arrow
module load cuda/12
source $SCRATCH/env/ds2d/bin/activate

TEST_RANGE=${1:-"1,100"}

python src/pred/run_generation.py \
    --with_feedback \
    --test_range "$TEST_RANGE" \
    --batch_size 2
