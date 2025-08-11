import os
from typing import Dict, List, Tuple, Optional
from src.utils.json_check.verify import is_valid_json
from src.metrics_v2.numerical.utils import NumericalUtils
from src.metrics_v2.numerical.calculator import NumericalMetricsCalculator


class NumericalEvaluate:
    """Aggregates numerical metrics across a results folder (overall stats, not split by room count)."""

    def __init__(self, folder_path: str, viz_round: int = 2):
        self.folder_path = folder_path
        self.viz_round = max(0, int(viz_round))
        self.metric_keys = [
            ("json_validity", "JSON Validity ↑"),
            ("room_count_match_pct", "Room Count ↑"),
            ("total_area_pct_diff", "Total Area ↑"),
            ("polygon_area_pct_diff_mean", "Polygon Area ↑"),
            ("overlap_present_pct", "Overlap ↓"),
            ("percentage_overlap_pct", "Percentage Overlap ↓"),
            ("prompt_room_count_pct", "Prompt Room Count ↑"),
            ("prompt_total_area_coverage_pct", "Prompt Total Area ↑"),
            ("prompt_room_id_recall_pct", "Prompt Room ID ↑"),
        ]

    def evaluate(self) -> Tuple[Dict[str, Tuple[Optional[float], Optional[float]]], List[int]]:
        # Collect sample metrics overall
        samples: Dict[str, List[Optional[float]]] = {k: [] for k, _ in self.metric_keys}
        valid_indices: List[int] = []

        for folder_name in sorted(os.listdir(self.folder_path), key=lambda x: int(x) if x.isdigit() else x):
            if not folder_name.isdigit():
                continue
            idx = int(folder_name)
            subfolder = os.path.join(self.folder_path, folder_name)
            if not os.path.isdir(subfolder):
                continue

            output_fp = NumericalUtils.load_json(os.path.join(subfolder, "0.json"))
            prompt_fp = NumericalUtils.load_json(os.path.join(subfolder, "prompt.json"))

            # record JSON validity for every sample
            is_valid = is_valid_json(output_fp) if isinstance(output_fp, dict) else False
            samples["json_validity"].append(1.0 if is_valid else 0.0)

            if not is_valid:
                continue

            sm = NumericalMetricsCalculator(output_fp, prompt_fp).compute()
            valid_indices.append(idx)
            for key, _title in self.metric_keys:
                if key == "json_validity":
                    continue
                samples[key].append(getattr(sm, key))

        # Aggregate mean/std overall
        stats: Dict[str, Tuple[Optional[float], Optional[float]]] = {}
        for key, _title in self.metric_keys:
            mean, std = NumericalUtils.mean_std(samples[key])
            stats[key] = (None if mean is None else round(mean, 4), None if std is None else round(std, 4))

        # Print Markdown table: single overall column
        header = "| Metric | mean ± std |"
        divider = "|------------|------------|"
        print(header)
        print(divider)
        for key, title in self.metric_keys:
            mean, std = stats[key]
            if mean is None:
                cell = "–"
            else:
                if key in ("total_area_pct_diff", "polygon_area_pct_diff_mean"):
                    # Display as ratio consistency: (1 - fraction_diff) in [0,1]
                    mean_disp = max(0.0, min(1.0, (1.0 - mean)))
                    std_disp = min(1.0, std if std is not None else 0.0)
                    cell = f"{mean_disp:.{self.viz_round}f} ± {std_disp:.{self.viz_round}f}"
                else:
                    cell = f"{mean:.{self.viz_round}f} ± {std:.{self.viz_round}f}"
            print(f"| {title} | {cell} |")
        print()

        return stats, valid_indices 