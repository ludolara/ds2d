from dataclasses import dataclass, field
from typing import Any, Dict, List, Set
from pathlib import Path
import json
from datasets import DatasetDict, Dataset
from sklearn.model_selection import train_test_split
from shapely.geometry import LineString, Polygon
from shapely.ops import polygonize
from shapely.affinity import scale
from tqdm import tqdm
from dataset_convert.rplan_graph import RPLANGraph
from utils.constants import RPLAN_ROOM_CLASS
from collections import Counter, defaultdict, OrderedDict
import random

@dataclass
class RPLANConverter:
    """
    Convert raw RPLAN JSON into a Hugging Face DatasetDict, including room polygons and adjacency.
    """
    round_value: int = 1
    original_map = RPLAN_ROOM_CLASS
    room_map: Dict[int, str] = field(init=False)
    pixel_to_meter: float = field(init=False)

    def __post_init__(self):
        # reverse original_map to map code → name
        self.room_map = {code: name for name, code in self.original_map.items()}
        # 18m × 18m house over 256px
        self.pixel_to_meter = 18 / 256

    def _map_room_type(self, code: int) -> str:
        return self.room_map.get(code, str(code))

    def _segments_to_polygon(self, segments: List[List[float]]):
        unique = {tuple(seg) for seg in segments}
        lines = [LineString([(x1, y1), (x2, y2)]) for x1, y1, x2, y2 in unique]
        polys = list(polygonize(lines))
        if not polys:
            raise ValueError("No polygon could be formed from segments.")
        if len(polys) > 1:
            raise ValueError(f"Multiple polygons detected: {len(polys)}")
        return polys[0] 

    def _shuffle_spaces(self, spaces: List[OrderedDict], seed: int = 42) -> List[Dict[str, Any]]:
        random.seed(seed)
        random_spaces = list(spaces)
        random.shuffle(random_spaces)
        return random_spaces

    def _convert_entry(self, data: Dict[str, Any]) -> Dict[str, Any]:
        counts = Counter(data["room_type"])
        idx_counters = defaultdict(int)

        temp = []
        for code in data["room_type"]:
            name = self._map_room_type(code)
            if counts[code] > 1:
                idx = idx_counters[code]
                room_id = f"{name}|{idx}"
                idx_counters[code] += 1
            else:
                room_id = name
            temp.append({
                "id": room_id,
                "room_type": name,
                "segments": []
            })

        # assign each edge segment to its first room
        for edge, rm_idxs in zip(data["edges"], data["ed_rm"]):
            seg = edge[:4]
            if rm_idxs:
                temp[rm_idxs[0]]["segments"].append(seg)

        total_area = 0.0
        spaces = []
        for room in temp:
            segs = room.pop("segments", [])
            if not segs:
                return None
            try:
                poly = self._segments_to_polygon(segs)
                poly = scale(poly, xfact=self.pixel_to_meter, yfact=self.pixel_to_meter, origin=(0,0))
                area = round(poly.area, self.round_value)
                minx, miny, maxx, maxy = poly.bounds
                spaces.append(OrderedDict({
                    **room,
                    "area": area,
                    "width": round(maxx - minx, self.round_value),
                    "height": round(maxy - miny, self.round_value),
                    "is_rectangular": 1 if (len(poly.exterior.coords) - 1) == 4 else 0,
                    "floor_polygon": [
                        {"x": round(x, self.round_value), "y": round(y, self.round_value)}
                        for x, y in poly.exterior.coords[:-1]
                    ]
                }))
                total_area += area
            except Exception as e:
                print(f"Error processing segments for room {room['id']}: {e}")
                return None

        # build index-based adjacency with RPLANGraph
        fp_graph = RPLANGraph({
            "room_type": data["room_type"],
            "ed_rm": data["ed_rm"]
        })
        bubble_diagram = fp_graph.bubble_diagram
        # remove entries with rooms with empty arraay
        if any(not neigh for neigh in bubble_diagram.values()):
            return None
        
        random_spaces = self._shuffle_spaces(spaces)

        return {
            "rplan_id": data.get("rplan_id"),
            "room_count": len(spaces),
            # "total_area": round(total_area, self.round_value),
            # "room_types": [r["room_type"] for r in spaces],
            "bubble_diagram": json.dumps(bubble_diagram),
            "rooms": random_spaces
        }

    def create_dataset(self, raw: List[Dict[str, Any]]) -> DatasetDict:
        converted = [
            out for item in tqdm(raw, desc="Converting entries")
            if (out := self._convert_entry(item)) is not None
        ]

        train, temp = train_test_split(converted, test_size=0.2, shuffle=False)
        test, val = train_test_split(temp, test_size=0.5, shuffle=False)

        return DatasetDict({
            "train": Dataset.from_list(train),
            "test": Dataset.from_list(test),
            "validation": Dataset.from_list(val),
        })

    def _load_folder(self, folder: str) -> List[Dict[str, Any]]:
        files = sorted(Path(folder).glob("*.json"), key=lambda p: int(p.stem))
        raw = []
        for p in tqdm(files, desc="Loading JSON files"):
            data = json.loads(p.read_text())
            data["rplan_id"] = p.stem
            raw.append(data)
        return raw

    def __call__(self, folder: str) -> DatasetDict:
        raw = self._load_folder(folder)
        return self.create_dataset(raw)

if __name__ == "__main__":
    converter = RPLANConverter()
    ds = converter("datasets/rplan_json_test")
    ds.save_to_disk("datasets/rplan_converted")
    print(ds)
    print("Done. Sample:", ds["train"][0])
