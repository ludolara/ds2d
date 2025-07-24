"""
Fair comparison of multiple model results by computing compatibility scores
only on the intersection of valid JSON instances across all models.
"""

import os
import sys
from eval_overall import Evaluate

def get_intersection_of_valid_indices(result_folders, room_counts):
    """
    Get the intersection of valid JSON indices across all result folders.
    Returns a dict mapping room_count -> list of valid indices common to all folders.
    """
    all_valid_indices = {}
    
    # Initialize with indices from first folder
    first_folder = result_folders[0]
    if not os.path.exists(first_folder):
        print(f"Warning: {first_folder} not found")
        return {rc: [] for rc in room_counts}
        
    ev = Evaluate(folder_path=first_folder, room_counts=room_counts)
    for rc in room_counts:
        all_valid_indices[rc] = set(ev.get_valid_indices_for(rc))
    
    # Intersect with indices from remaining folders
    for folder in result_folders[1:]:
        if not os.path.exists(folder):
            print(f"Warning: {folder} not found, skipping")
            continue
            
        ev = Evaluate(folder_path=folder, room_counts=room_counts)
        for rc in room_counts:
            folder_valid_indices = set(ev.get_valid_indices_for(rc))
            all_valid_indices[rc] = all_valid_indices[rc].intersection(folder_valid_indices)
    
    # Convert back to sorted lists
    return {rc: sorted(list(indices)) for rc, indices in all_valid_indices.items()}

def compare_models(result_folders, model_names=None, room_counts=None):
    """
    Compare multiple models fairly by computing scores on intersection of valid instances.
    
    Args:
        result_folders: List of paths to result folders
        model_names: List of model names (defaults to folder names)
        room_counts: List of room counts to evaluate (defaults to [5,6,7,8])
    """
    if room_counts is None:
        room_counts = [5, 6, 7, 8]
        
    if model_names is None:
        model_names = [folder for folder in result_folders]
    
    common_valid_indices = get_intersection_of_valid_indices(result_folders, room_counts)
    
    # Collect stats using all valid instances for each model
    all_stats_union = {}
    for folder, model_name in zip(result_folders, model_names):
        if os.path.exists(folder):
            ev = Evaluate(folder_path=folder, room_counts=room_counts)
            stats, _ = ev.evaluate(valid_indices=None)
            all_stats_union[model_name] = stats
    
    # Collect stats using intersection of valid instances
    all_stats_intersection = {}
    for folder, model_name in zip(result_folders, model_names):
        if os.path.exists(folder):
            ev = Evaluate(folder_path=folder, room_counts=room_counts)
            stats, _ = ev.evaluate(valid_indices=common_valid_indices)
            all_stats_intersection[model_name] = stats
    
    # Print summary
    print(f"Intersection uses {len(common_valid_indices.get(8, []))} common instances")
    
    # Header
    header_cells = ["Model"]
    for rc in room_counts:
        header_cells.extend([f"{rc} spaces", f"{rc} error %"])
    
    header = "| " + " | ".join(header_cells) + " |"
    divider = "|" + "|".join(["------------"] * len(header_cells)) + "|"
    
    # Print union comparison table
    print(f"\n{'='*60}")
    print("UNION: Each model evaluated on its own valid instances")
    print(f"{'='*60}")
    
    print(header)
    print(divider)
    
    for model_name in model_names:
        if model_name not in all_stats_union:
            continue
        row_cells = [model_name]
        stats = all_stats_union[model_name]
        for rc in room_counts:
            if rc in stats and stats[rc][0] is not None:
                row_cells.append(f"{stats[rc][0]:.2f} ± {stats[rc][1]:.2f}")
                row_cells.append(f"{stats[rc][2]:.1f}%")
            else:
                row_cells.extend(["–", "–"])
        row = "| " + " | ".join(row_cells) + " |"
        print(row)
    
    # Print intersection comparison table
    print(f"\n{'='*60}")
    print("INTERSECTION: All models evaluated on same valid instances")
    print(f"{'='*60}")
    
    print(header)
    print(divider)
    
    for model_name in model_names:
        if model_name not in all_stats_intersection:
            continue
        row_cells = [model_name]
        stats = all_stats_intersection[model_name]
        for rc in room_counts:
            if rc in stats and stats[rc][0] is not None:
                row_cells.append(f"{stats[rc][0]:.2f} ± {stats[rc][1]:.2f}")
                row_cells.append(f"{stats[rc][2]:.1f}%")
            else:
                row_cells.extend(["–", "–"])
        row = "| " + " | ".join(row_cells) + " |"
        print(row)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python compare_models.py <result_folder1> [result_folder2] [result_folder3] ...")
        print("Example: python compare_models.py results_model1/ results_model2/ results_model3/")
        sys.exit(1)
    
    result_folders = sys.argv[1:]
    
    # Auto-detect model names from folder paths
    model_names = []
    for folder in result_folders:
        # name = os.path.basename(folder.rstrip('/'))
        # if name.startswith('results_'):
        #     name = name[8:]  # Remove 'results_' prefix
        model_names.append(folder)
    
    compare_models(result_folders, model_names) 