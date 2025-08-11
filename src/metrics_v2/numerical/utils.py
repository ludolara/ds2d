from typing import Dict, List, Optional, Tuple
from shapely.geometry import Polygon as ShapelyPolygon
from src.utils.constants import OVERLAP_TOL
import json


class NumericalUtils:
    DOOR_TYPES = {"interior_door"}

    @staticmethod
    def get_rooms_list(fp: dict) -> List[dict]:
        rooms = fp.get("spaces")
        if rooms is None:
            rooms = fp.get("rooms", [])
        return rooms if isinstance(rooms, list) else []

    @staticmethod
    def is_door(room: dict) -> bool:
        return str(room.get("room_type", "")).lower() in NumericalUtils.DOOR_TYPES

    @staticmethod
    def load_json(path: str) -> Optional[dict]:
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return None

    @staticmethod
    def mean_std(values: List[Optional[float]]) -> Tuple[Optional[float], Optional[float]]:
        import statistics
        clean = [v for v in values if v is not None]
        if not clean:
            return None, None
        if len(clean) == 1:
            return clean[0], 0.0
        return statistics.mean(clean), statistics.stdev(clean)

    @staticmethod
    def percent_diff(a: Optional[float], b: Optional[float]) -> Optional[float]:
        try:
            if a is None or b is None:
                return None
            if a == 0:
                return None
            return abs(a - b) / abs(a) * 100.0
        except Exception:
            return None

    @staticmethod
    def safe_ratio(numer: Optional[float], denom: Optional[float]) -> Optional[float]:
        try:
            if numer is None or denom is None or denom == 0:
                return None
            return float(numer) / float(denom)
        except Exception:
            return None

    @staticmethod
    def room_ids_set_from_spaces(spaces: List[dict]) -> set:
        ids = set()
        for room in spaces:
            try:
                if NumericalUtils.is_door(room):
                    continue
                rid = room.get("id")
                if rid is None:
                    continue
                rid = str(rid)
                if "Placeholder" in rid:
                    continue
                ids.add(rid)
            except Exception:
                continue
        return ids

    @staticmethod
    def compute_actual_room_count(output_fp: dict) -> Optional[int]:
        try:
            rooms = [r for r in NumericalUtils.get_rooms_list(output_fp) if not NumericalUtils.is_door(r)]
            return max(0, len(rooms) - 1)
        except Exception:
            return None

    @staticmethod
    def extract_polygons_and_areas(output_fp: dict) -> Tuple[Dict[str, ShapelyPolygon], Dict[str, float]]:
        polygons: Dict[str, ShapelyPolygon] = {}
        areas: Dict[str, float] = {}
        for idx, room in enumerate(NumericalUtils.get_rooms_list(output_fp)):
            try:
                if NumericalUtils.is_door(room):
                    continue
                room_id = str(room.get("id", idx))
                pts = room.get("floor_polygon", [])
                coords = [(float(p["x"]), float(p["y"])) for p in pts]
                if len(coords) < 3:
                    continue
                poly = ShapelyPolygon(coords)
                if not poly.is_valid:
                    poly = poly.buffer(0)
                if not poly.is_valid or poly.area <= OVERLAP_TOL:
                    continue
                key = room_id if room_id not in polygons else f"{room_id}_{idx}"
                polygons[key] = poly
                if isinstance(room.get("area"), (int, float)):
                    areas[key] = float(room["area"])
            except Exception:
                continue
        return polygons, areas

    @staticmethod
    def compute_overlap_stats(polygons: Dict[str, ShapelyPolygon]) -> Tuple[bool, float]:
        filtered = {k: v for k, v in polygons.items() if "interior_door" not in str(k).lower() and "front_door" not in str(k).lower()}
        keys = list(filtered.keys())
        total_overlap = 0.0
        has_overlap = False
        for i in range(len(keys)):
            p1 = filtered[keys[i]]
            for j in range(i + 1, len(keys)):
                p2 = filtered[keys[j]]
                if p1.intersects(p2):
                    inter = p1.intersection(p2)
                    if not inter.is_empty and inter.area > OVERLAP_TOL:
                        total_overlap += inter.area
                        has_overlap = True
        return has_overlap, total_overlap 