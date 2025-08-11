import os
import json
import argparse
from src.metrics_v2.numerical.evaluator import NumericalEvaluate


def main():
    parser = argparse.ArgumentParser(description="Numerical metrics aggregator")
    parser.add_argument("results_folder", type=str, help="Path to results folder (e.g., results/5_6_7_8)")
    parser.add_argument("--round", dest="viz_round", type=int, default=2, help="Rounding precision for displayed numbers (default: 2)")
    args = parser.parse_args()

    eval_path = args.results_folder
    evaluator = NumericalEvaluate(eval_path, viz_round=args.viz_round)
    stats, valid_indices = evaluator.evaluate()

    result_folder = eval_path.split('/')[0]
    output_dir = f"final_results/{result_folder}"
    os.makedirs(output_dir, exist_ok=True)

    payload = {
        "model_name": "DS2D v2",
        "eval_path": eval_path,
        "stats": {k: {"mean": stats[k][0], "std": stats[k][1]} for k, _ in evaluator.metric_keys},
        "valid_indices": valid_indices,
    }

    with open(os.path.join(output_dir, "numerical.json"), "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=4)


if __name__ == "__main__":
    main()
