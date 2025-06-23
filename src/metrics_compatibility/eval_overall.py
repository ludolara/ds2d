import os
import json
import statistics
from src.dataset_convert.rplan_graph import RPLANGraph

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
        Returns (mean, stdev) or (None, None) if no cases found.
        """
        scores = []
        for folder_name in sorted(os.listdir(self.folder_path),
                                  key=lambda x: int(x) if x.isdigit() else x):
            subfolder = os.path.join(self.folder_path, folder_name)
            if not os.path.isdir(subfolder):
                continue
            try:
                with open(os.path.join(subfolder, 'prompt.json')) as pf:
                    prompt = json.load(pf)
                if prompt.get("room_count") != rc:
                    continue
                with open(os.path.join(subfolder, '0.json')) as of:
                    output = json.load(of)

                input_graph    = RPLANGraph.from_ds2d(output)
                expected_graph = RPLANGraph.from_labeled_adjacency(
                    prompt["input_graph"]
                )
                scores.append(
                    input_graph.compatibility_score(expected_graph)
                )
            except Exception as e:
                print(f"Error in folder {folder_name} (rc={rc}): {e}")

        if not scores:
            return None, None
        mean  = statistics.mean(scores)
        stdev = statistics.stdev(scores) if len(scores) > 1 else 0.0
        return mean, stdev

    def evaluate(self):
        """
        Compute raw compatibility stats for each room_count in self.room_counts
        and print a Markdown table (Model: DS2D v2).
        """
        # Gather stats
        stats = {}
        for rc in self.room_counts:
            mean, stdev = self._compute_raw_for(rc)
            if mean is not None:
                stats[rc] = (mean, stdev)

        # Print Markdown table
        header_cells = ["Model"] + [f"{rc} rooms" for rc in self.room_counts]
        header = "| " + " | ".join(header_cells) + " |"
        divider = "|" + "|".join(["------------"] * len(header_cells)) + "|"

        row_cells = ["DS2D v2"] + [
            f"{stats[rc][0]:.2f} ± {stats[rc][1]:.2f}" if rc in stats else "–"
            for rc in self.room_counts
        ]
        row = "| " + " | ".join(row_cells) + " |"

        print(header)
        print(divider)
        print(row)

        return stats


if __name__ == "__main__":
    ev = Evaluate(folder_path="results/")
    ev.evaluate()
