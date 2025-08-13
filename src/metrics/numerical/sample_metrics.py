from dataclasses import dataclass
from typing import Optional


@dataclass
class SampleMetrics:
    room_count_match_pct: Optional[float]
    total_area_pct_diff: Optional[float]
    polygon_area_pct_diff_mean: Optional[float]
    overlap_present_pct: Optional[float]
    percentage_overlap_pct: Optional[float]
    prompt_room_count_pct: Optional[float]
    prompt_total_area_coverage_pct: Optional[float]
    prompt_room_id_recall_pct: Optional[float]
    prompt_room_area_compliance_pct: Optional[float]
    prompt_room_area_mape_pct: Optional[float] 