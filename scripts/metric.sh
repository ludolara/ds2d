export PYTHONPATH="$PYTHONPATH:/."
module load python/3.11
source $SCRATCH/env/vllm/bin/activate

RESULT_FOLDER=${1:-"results/results5_GRPO_70B_fs/"}

echo "Using result folder: $RESULT_FOLDER"

python src/metrics/numerical/run_metric.py "$RESULT_FOLDER"
python src/metrics/compatibility/run_metric.py "$RESULT_FOLDER"

source $SCRATCH/env/hd/bin/activate
python src/metrics/diversity/run_metric.py "$RESULT_FOLDER" "direct"
