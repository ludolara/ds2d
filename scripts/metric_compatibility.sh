export PYTHONPATH="$PYTHONPATH:/."
module load python/3.11
source $SCRATCH/env/vllm/bin/activate

RESULT_FOLDER=${1:-"results_70B_r256_GRPO_sampling/generations/rplan_8_70B/full_prompt"}

python src/metrics_compatibility/run_metrics.py "$RESULT_FOLDER"
