from typing import Dict, List, Optional
from src.metrics_v2.numerical.utils import NumericalUtils
from src.metrics_v2.numerical.sample_metrics import SampleMetrics


class NumericalMetricsCalculator:
    """Computes numerical consistency metrics for a single sample (output + prompt)."""

    def __init__(self, output_fp: dict, prompt_fp: Optional[dict]):
        self.output_fp = output_fp
        self.prompt_fp = prompt_fp or {}
        self.polygons, self.stated_areas = NumericalUtils.extract_polygons_and_areas(output_fp)
        self.actual_room_count = NumericalUtils.compute_actual_room_count(output_fp)

    def compute(self) -> SampleMetrics:
        # 1) Room Count match → 1 if exact, else 0
        gen_room_count_field = self.output_fp.get("room_count")
        if isinstance(gen_room_count_field, int) and self.actual_room_count is not None:
            room_count_match_pct = 1.0 if gen_room_count_field == self.actual_room_count else 0.0
        else:
            room_count_match_pct = None

        # 2) Total Area percent (↑) 
        total_area_field = self.output_fp.get("total_area")
        sum_stated_areas = sum(self.stated_areas.values()) if self.stated_areas else 0.0
        tad = NumericalUtils.percent_diff(float(total_area_field) if isinstance(total_area_field, (int, float)) else None,
                                          sum_stated_areas)
        total_area_pct_diff = None if tad is None else tad / 100.0

        # 3) Polygon Area (↑) 
        per_room_pds: List[float] = []
        for key, poly in self.polygons.items():
            stated_area = self.stated_areas.get(key)
            pr_pd = NumericalUtils.percent_diff(float(stated_area) if isinstance(stated_area, (int, float)) else None,
                                                poly.area)
            if pr_pd is not None:
                per_room_pds.append(pr_pd / 100.0)
        polygon_area_pct_diff_mean = (sum(per_room_pds) / len(per_room_pds)) if per_room_pds else None

        # 4) Overlap present (↓) and 5) Percentage Overlap (↓)
        has_overlap, total_overlap_area = NumericalUtils.compute_overlap_stats(self.polygons)
        overlap_present_pct = 1.0 if has_overlap else 0.0
        denom_area_for_overlap = (float(total_area_field)
                                  if isinstance(total_area_field, (int, float)) and total_area_field > 0
                                  else sum(p.area for p in self.polygons.values()))
        percentage_overlap_pct = NumericalUtils.safe_ratio(total_overlap_area, denom_area_for_overlap)

        # Prompt-based metrics (all as ratios in [0,1])
        prompt_room_count_pct: Optional[float] = None
        prompt_total_area_coverage_pct: Optional[float] = None
        prompt_room_id_recall_pct: Optional[float] = None

        if isinstance(self.prompt_fp, dict):
            # 6) Prompt Room Count (↑): min(actual, prompt)/prompt
            prompt_room_count = self.prompt_fp.get("room_count")
            if isinstance(prompt_room_count, int) and prompt_room_count > 0 and self.actual_room_count is not None:
                prompt_room_count_pct = min(self.actual_room_count, prompt_room_count) / prompt_room_count

            # 7) Prompt Total Area coverage (↑): sum polygon areas / prompt_total_area
            prompt_total_area = self.prompt_fp.get("total_area")
            if isinstance(prompt_total_area, (int, float)) and prompt_total_area > 0:
                polygon_total_area = sum(p.area for p in self.polygons.values())
                cov = NumericalUtils.safe_ratio(polygon_total_area, float(prompt_total_area))
                prompt_total_area_coverage_pct = cov

            # 8) Prompt Room ID recall (↑): |gen ∩ prompt| / |prompt|
            gen_ids = NumericalUtils.room_ids_set_from_spaces(NumericalUtils.get_rooms_list(self.output_fp))
            prompt_ids = NumericalUtils.room_ids_set_from_spaces(self.prompt_fp.get("spaces", []))
            if len(prompt_ids) > 0:
                prompt_room_id_recall_pct = len(gen_ids & prompt_ids) / len(prompt_ids)

        return SampleMetrics(
            room_count_match_pct=room_count_match_pct,
            total_area_pct_diff=total_area_pct_diff,
            polygon_area_pct_diff_mean=polygon_area_pct_diff_mean,
            overlap_present_pct=overlap_present_pct,
            percentage_overlap_pct=percentage_overlap_pct,
            prompt_room_count_pct=prompt_room_count_pct,
            prompt_total_area_coverage_pct=prompt_total_area_coverage_pct,
            prompt_room_id_recall_pct=prompt_room_id_recall_pct,
        ) 