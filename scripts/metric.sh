export PYTHONPATH="$PYTHONPATH:/."
module load python/3.11
source $SCRATCH/env/vllm/bin/activate

FULL_RESULT_FOLDER=${1:-"results8_70B/generations/rplan_8_70B/full_prompt"}
RESULT_FOLDER=$(echo "$FULL_RESULT_FOLDER" | cut -d'/' -f1)

echo "Using result folder: $RESULT_FOLDER"

python src/metrics/run_metrics.py "$RESULT_FOLDER"
python src/metrics_v2/compatibility/run_metric.py "$FULL_RESULT_FOLDER"

source $SCRATCH/env/hd/bin/activate
python src/metrics_v2/diversity/run_metric.py "$RESULT_FOLDER"
