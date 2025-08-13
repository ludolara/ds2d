import json
from dataclasses import dataclass, field
import random
from typing import Any, Dict, List
from pathlib import Path
from datasets import DatasetDict, Dataset
from sklearn.model_selection import train_test_split
from shapely.geometry import LineString
from shapely.ops import polygonize
from shapely.affinity import scale
from tqdm import tqdm
from dataset_convert.rplan_graph import RPLANGraph
from utils.constants import RPLAN_ROOM_CLASS
from collections import Counter, defaultdict
import networkx as nx

@dataclass
class RPLANConverter:
    """
    Convert raw RPLAN housegan JSON into a Hugging Face DatasetDict.
    """
    round_value: int = 2
    original_map = RPLAN_ROOM_CLASS
    room_map: Dict[int, str] = field(init=False)
    pixel_to_meter: float = field(init=False)
    room_number: int = 8

    def __post_init__(self):
        # reverse original_map to map code → name
        self.room_map = {code: name for name, code in self.original_map.items()}
        # 18m × 18m house over 256px
        # self.pixel_to_meter = 18 / 256

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

        # total_area = 0.0
        total_area = 0
        spaces = []
        for room in temp:
            segs = room.pop("segments", [])
            if not segs:
                return None
            try:
                poly = self._segments_to_polygon(segs)
                # poly = scale(poly, xfact=self.pixel_to_meter, yfact=self.pixel_to_meter, origin=(0,0))
                # area = round(poly.area, self.round_value)
                # minx, miny, maxx, maxy = poly.bounds
                # is_rectangular = (len(poly.exterior.coords) - 1) == 4

                # room_data = {
                #     **room,
                #     "area": area,
                #     "width": round(maxx - minx, self.round_value) if is_rectangular else 0,
                #     "height": round(maxy - miny, self.round_value) if is_rectangular else 0,
                #     "floor_polygon": [
                #         {"x": round(x, self.round_value), "y": round(y, self.round_value)}
                #         for x, y in poly.exterior.coords[:-1]
                #     ]
                # }
                area = int(poly.area)
                minx, miny, maxx, maxy = int(poly.bounds[0]), int(poly.bounds[1]), int(poly.bounds[2]), int(poly.bounds[3])
                is_rectangular = (len(poly.exterior.coords) - 1) == 4

                room_data = {
                    **room,
                    "area": area,
                    "width": int(maxx - minx) if is_rectangular else 0,
                    "height": int(maxy - miny) if is_rectangular else 0,
                    "floor_polygon": [
                        {"x": int(x), "y": int(y)}
                        for x, y in poly.exterior.coords[:-1]
                    ]
                }
                
                spaces.append(room_data)
                if room["room_type"] not in ["interior_door", "front_door"]:
                    total_area += area
            except Exception as e:
                print(f"Error processing segments for room {room['id']}: {e}")
                return None

        # build index-based adjacency with RPLANGraph
        fp_graph = RPLANGraph.from_housegan({
            "room_type": data["room_type"],
            "ed_rm": data["ed_rm"]
        })
        input_graph = fp_graph.to_labeled_adjacency()
        # remove entries with spaces with empty array
        if any(not neigh for neigh in input_graph.values()) or not nx.is_connected(nx.from_dict_of_lists(input_graph)):
            return None
        
        only_rooms = [r for r in spaces if r["room_type"] not in ["interior_door"]]
        input_rooms = []

        # for room in spaces:
        for room in only_rooms:
            if room["height"] == 0 and room["width"] == 0:
                input_room = {
                    "id": room["id"],
                    "room_type": room["room_type"],
                    "area": room["area"]
                }
            else:
                input_room = {
                    "id": room["id"],
                    "room_type": room["room_type"],
                    "height": room["height"],
                    "width": room["width"]
                }
            input_rooms.append(input_room)

        # create input
        input_data = {
            "input": {
                "room_count": len(only_rooms) - 1,
                # "total_area": round(total_area, self.round_value),
                "total_area": int(total_area),
                "spaces": input_rooms,
                "input_graph": input_graph
            }
        }

        return {
            "rplan_id": data.get("rplan_id"),
            "room_count": len(only_rooms) - 1,
            # "total_area": round(total_area, self.round_value),
            "total_area": int(total_area),
            "input_graph": json.dumps(input_graph),
            "spaces": spaces,
            "prompt": json.dumps(input_data)
        }

    def create_dataset(self, raw: List[Dict[str, Any]]) -> DatasetDict:
        converted = [
            out for item in tqdm(raw, desc="Converting entries")
            if (out := self._convert_entry(item)) is not None
        ]

        if self.room_number == 0:
            train, test_val = train_test_split(converted, test_size=0.2, shuffle=True)
            test, val = train_test_split(test_val, test_size=0.5, shuffle=True)
        else:
            target_room_plans = [item for item in converted if item["room_count"] == self.room_number]
            other_plans = [item for item in converted if item["room_count"] != self.room_number]
            test, val = train_test_split(target_room_plans, test_size=0.5, shuffle=True)
            random.seed(84)
            random.shuffle(other_plans)
            train = other_plans

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
    for room_number in [5, 6, 7, 8]:
        print(f"Processing rplan_{room_number}")
        converter = RPLANConverter(room_number=room_number)
        ds = converter("datasets/rplan_json")
        ds.save_to_disk(f"datasets/final_2/rplan_{room_number}")
        print(f"Saved rplan_{room_number}")
        print(ds)
        print("Train sample:", ds["train"][0])
        print("Test sample:", ds["test"][0])
        print("Val sample:", ds["validation"][0])
