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

    def _compute_raw_for(self, rc):
        """
        Compute mean and std dev of raw compatibility for room_count == rc.
        Returns (mean, stdev, error_rate) or (None, None, None) if no cases found.
        """
        scores = []
        target_attempts = 0  # Attempts that were trying to generate rc rooms
        successful_attempts = 0  # Attempts that actually generated rc rooms
        
        for folder_name in sorted(os.listdir(self.folder_path),
                                  key=lambda x: int(x) if x.isdigit() else x):
            subfolder = os.path.join(self.folder_path, folder_name)
            if not os.path.isdir(subfolder):
                continue
            try:
                with open(os.path.join(subfolder, 'prompt.json')) as pf:
                    prompt = json.load(pf)

                # Check if this prompt was trying to generate rc rooms
                expected_room_count = prompt.get("room_count")
                if expected_room_count != rc:
                    continue
                
                target_attempts += 1

                with open(os.path.join(subfolder, '0.json')) as of:
                    output = json.load(of)
                
                if not is_valid_json(output):
                    continue

                input_graph = RPLANGraph.from_ds2d(output)
                
                rooms = output.get('rooms', [])
                door_types = {'front_door', 'interior_door'}
                actual_room_count = len([room for room in rooms if room.get('room_type', '').lower() not in door_types])
                if actual_room_count != rc:
                    continue
                
                successful_attempts += 1
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
            return None, None, None
            
        error_rate = ((target_attempts - successful_attempts) / target_attempts * 100) if target_attempts > 0 else 0
        
        if not scores:
            return None, None, error_rate
            
        mean  = statistics.mean(scores)
        stdev = statistics.stdev(scores) if len(scores) > 1 else 0.0
        return mean, stdev, error_rate

    def evaluate(self):
        """
        Compute raw compatibility stats for each room_count in self.room_counts
        and print a Markdown table (Model: DS2D v2).
        """
        # Gather stats
        stats = {}
        for rc in self.room_counts:
            mean, stdev, error_rate = self._compute_raw_for(rc)
            if mean is not None or error_rate is not None:
                stats[rc] = (mean, stdev, error_rate)

        # Print Markdown table
        header_cells = ["Model"]
        for rc in self.room_counts:
            header_cells.extend([f"{rc} rooms", f"{rc} error %"])
        
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

        return stats


if __name__ == "__main__":
    ev = Evaluate(folder_path="results/")
    ev.evaluate()
