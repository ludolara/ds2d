import sys
eval_path = sys.argv[1]

from src.metrics_compatibility.eval_overall import Evaluate

overall_evaluation = Evaluate(eval_path)

overall_evaluation.evaluate()
