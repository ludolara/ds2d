export PYTHONPATH="$PYTHONPATH:/."
module load python/3.11
source $SCRATCH/env/vllm/bin/activate

RESULT_FOLDER=${1:-"results_70B_r512/"}

python src/metrics/run_metrics.py "$RESULT_FOLDER"
