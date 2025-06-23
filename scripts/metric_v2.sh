export PYTHONPATH="$PYTHONPATH:/."
module load python/3.11
source $SCRATCH/env/vllm/bin/activate

RESULT_FOLDER=${1:-"results_70B_r256_GRPO_7n_cp1100/generations/rplan_7_70B/full_prompt"}

python src/metrics_v2/run_metrics.py "$RESULT_FOLDER"
