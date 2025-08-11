import sys
import os
import json

eval_path = sys.argv[1]

from src.metrics_v2.compatibility.eval_overall import Evaluate

overall_evaluation = Evaluate(eval_path)

stats, all_valid_indices = overall_evaluation.evaluate()

result_folder = eval_path.split('/')[0]

output_dir = f"final_results/{result_folder}"
os.makedirs(output_dir, exist_ok=True)

json_data = {
    "model_name": "DS2D v2",
    "eval_path": eval_path,
    "room_counts": overall_evaluation.room_counts,
    "stats": {}
}

for rc in overall_evaluation.room_counts:
    if rc in stats and stats[rc][0] is not None:
        json_data["stats"][str(rc)] = {
            "mean": round(stats[rc][0], 2),
            "std": round(stats[rc][1], 2), 
            "error_percentage": round(stats[rc][2], 2)
        }
    else:
        json_data["stats"][str(rc)] = None

# Save to file
output_file = f"{output_dir}/compatibility.json"
with open(output_file, 'w') as f:
    json.dump(json_data, f, indent=4)
