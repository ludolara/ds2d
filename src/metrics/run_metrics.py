import sys
import os
import json

eval_path = sys.argv[1]

from src.metrics.eval_overall import Evaluate

overall_evaluation = Evaluate(eval_path,
                              metrics='all',
                              experiment_list='all',
                              if_separate_num_room_results=False)

# Get results from evaluation
results = overall_evaluation.evaluate_aggregate()

# Extract the first folder name (before the first slash)
result_folder = eval_path.split('/')[0] if '/' in eval_path else eval_path

# Create output directory
output_dir = f"final_results/{result_folder}"
os.makedirs(output_dir, exist_ok=True)

# Save to JSON file
output_file = f"{output_dir}/metrics.json"
with open(output_file, 'w') as f:
    json.dump(results, f, indent=4)
