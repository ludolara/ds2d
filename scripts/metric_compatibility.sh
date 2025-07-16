export PYTHONPATH="$PYTHONPATH:/."
module load python/3.11
source $SCRATCH/env/vllm/bin/activate

RESULT_FOLDER=${1:-"results_GRPO_70B_r128_a256_allu_sampling4/generations/rplan_8_70B/full_prompt"}

python src/metrics_compatibility/run_metrics.py "$RESULT_FOLDER"
