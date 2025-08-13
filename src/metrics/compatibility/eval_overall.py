import os
import json
import statistics
from src.dataset_convert.rplan_graph import RPLANGraph
from src.utils.json_check.verify import is_valid_json

class Evaluate:
    """
    Compute and print a Markdown table of raw compatibility scores
    for specified room counts.
    """
    def __init__(self, folder_path='results/', room_counts=None):
        self.folder_path = folder_path
        self.room_counts = room_counts or [5, 6, 7, 8]

    def get_valid_indices_for(self, rc, valid_indices=None):
        """
        Get the indices of instances that have valid JSON and meet all criteria for room_count == rc.
        If valid_indices is provided, only consider those indices.
        Returns a list of folder indices (as integers) that have valid JSON.
        """
        valid_folder_indices = []
        
        for folder_name in sorted(os.listdir(self.folder_path),
                                  key=lambda x: int(x) if x.isdigit() else x):
            if not folder_name.isdigit():
                continue
                
            folder_idx = int(folder_name)
            
            # If valid_indices is provided, only consider those indices
            if valid_indices is not None and folder_idx not in valid_indices:
                continue
                
            subfolder = os.path.join(self.folder_path, folder_name)
            if not os.path.isdir(subfolder):
                continue
                
            try:
                with open(os.path.join(subfolder, 'prompt.json')) as pf:
                    prompt = json.load(pf)

                # Check if this prompt was trying to generate rc spaces
                # expected_room_count = prompt.get("room_count")
                # if expected_room_count != rc:
                #     continue

                with open(os.path.join(subfolder, '0.json')) as of:
                    output = json.load(of)
                
                if not is_valid_json(output):
                    continue

                # input_graph = RPLANGraph.from_ds2d(output)
                
                spaces = output.get('spaces', [])
                door_types = {'interior_door'}
                actual_room_count = len([room for room in spaces if room.get('room_type', '').lower() not in door_types]) - 1
                if actual_room_count != rc:
                    continue
                
                # If we reach here, this instance is valid
                valid_folder_indices.append(folder_idx)
                
            except Exception as e:
                # print(f"Error in folder {folder_name} (rc={rc}): {e}")
                continue

        return valid_folder_indices

    def _compute_raw_for(self, rc, valid_indices=None):
        """
        Compute mean and std dev of raw compatibility for room_count == rc.
        If valid_indices is provided, only consider those indices.
        Returns (mean, stdev, error_rate, valid_indices_used) or (None, None, None, []) if no cases found.
        """
        scores = []
        target_attempts = 0  # Attempts that were trying to generate rc spaces
        successful_attempts = 0  # Attempts that actually generated rc spaces
        valid_indices_used = []
        
        for folder_name in sorted(os.listdir(self.folder_path),
                                  key=lambda x: int(x) if x.isdigit() else x):
            if not folder_name.isdigit():
                continue
                
            folder_idx = int(folder_name)
            
            # If valid_indices is provided, only consider those indices
            if valid_indices is not None and folder_idx not in valid_indices:
                continue
                
            subfolder = os.path.join(self.folder_path, folder_name)
            if not os.path.isdir(subfolder):
                continue
            try:
                with open(os.path.join(subfolder, 'prompt.json')) as pf:
                    prompt = json.load(pf)

                # Check if this prompt was trying to generate rc spaces
                # expected_room_count = prompt.get("room_count")
                # if expected_room_count != rc:
                #     continue
                
                target_attempts += 1

                with open(os.path.join(subfolder, '0.json')) as of:
                    output = json.load(of)
                
                if not is_valid_json(output):
                    continue

                input_graph = RPLANGraph.from_ds2d(output)
                
                spaces = output.get('spaces', [])
                door_types = {'interior_door'}
                actual_room_count = len([room for room in spaces if room.get('room_type', '').lower() not in door_types]) - 1
                if actual_room_count != rc:
                    continue
                
                successful_attempts += 1
                valid_indices_used.append(folder_idx)
                expected_graph = RPLANGraph.from_labeled_adjacency(
                    prompt["input_graph"]
                )
                scores.append(
                    input_graph.compatibility_score(expected_graph)
                )
            except Exception as e:
                # print(f"Error in folder {folder_name} (rc={rc}): {e}")
                continue

        if target_attempts == 0:
            return None, None, None, valid_indices_used
            
        error_rate = ((target_attempts - successful_attempts) / target_attempts * 100) if target_attempts > 0 else 0

        # print(f"target_attempts: {target_attempts}, successful_attempts: {successful_attempts}, error_rate: {error_rate}")  
        
        if not scores:
            return None, None, error_rate, valid_indices_used
            
        mean  = statistics.mean(scores)
        stdev = statistics.stdev(scores) if len(scores) > 1 else 0.0
        return mean, stdev, error_rate, valid_indices_used

    def evaluate(self, valid_indices=None):
        """
        Compute raw compatibility stats for each room_count in self.room_counts
        and print a Markdown table (Model: DS2D v2).
        If valid_indices is provided, only compute stats on those indices.
        Returns (stats, all_valid_indices) where all_valid_indices contains the indices 
        of valid JSON instances for each room count.
        """
        # Gather stats
        stats = {}
        all_valid_indices = {}
        
        for rc in self.room_counts:
            mean, stdev, error_rate, valid_indices_used = self._compute_raw_for(rc, valid_indices)
            if mean is not None or error_rate is not None:
                stats[rc] = (mean, stdev, error_rate)
            all_valid_indices[rc] = valid_indices_used

        # Print Markdown table
        header_cells = ["Model"]
        for rc in self.room_counts:
            header_cells.extend([f"{rc} spaces", f"{rc} error %"])
        
        header = "| " + " | ".join(header_cells) + " |"
        divider = "|" + "|".join(["------------"] * len(header_cells)) + "|"

        row_cells = ["DS2D v2"]
        for rc in self.room_counts:
            if rc in stats:
                if stats[rc][0] is not None:
                    row_cells.append(f"{stats[rc][0]:.2f} ± {stats[rc][1]:.2f}")
                else:
                    row_cells.append("–")
                row_cells.append(f"{stats[rc][2]:.1f}%")
            else:
                row_cells.extend(["–", "–"])
        
        row = "| " + " | ".join(row_cells) + " |"

        print(header)
        print(divider)
        print(row)

        return stats, all_valid_indices
