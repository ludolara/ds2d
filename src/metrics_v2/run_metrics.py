import sys
eval_path = sys.argv[1]

from src.metrics_v2.eval_overall import Evaluate

overall_evaluation = Evaluate(eval_path,
                              metrics='all')

overall_evaluation.evaluate_aggregate()
