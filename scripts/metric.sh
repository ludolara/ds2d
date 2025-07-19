export PYTHONPATH="$PYTHONPATH:/."
module load python/3.11
source $SCRATCH/env/vllm/bin/activate

FULL_RESULT_FOLDER=${1:-"results6_70B_r64_a128_all/generations/rplan_6_70B/full_prompt"}
RESULT_FOLDER=$(echo "$FULL_RESULT_FOLDER" | cut -d'/' -f1)

echo "Using result folder: $RESULT_FOLDER"

python src/metrics/run_metrics.py "$RESULT_FOLDER"
python src/metrics_compatibility/run_metrics.py "$FULL_RESULT_FOLDER"
