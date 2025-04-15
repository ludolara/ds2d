export PYTHONPATH="$PYTHONPATH:/."

module load python/3.11
module load arrow
source $SCRATCH/env/ds2d/bin/activate

python src/metrics/run_metrics.py results_feedback_edges/