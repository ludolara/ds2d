from typing import List, Dict, Any, Union
from pathlib import Path
import json
from datasets import DatasetDict, Dataset
from sklearn.model_selection import train_test_split
from shapely import Polygon
from shapely.geometry import LineString
from shapely.ops import polygonize

class RPLANConverter:
    def __init__(self, round_value: int = 1):
        original_map = {"living_room": 1, "kitchen": 2, "bedroom": 3, "bathroom": 4, "balcony": 5, "entrance": 6, "dining_room": 7, "study_room": 8, "storage": 10 , "front_door": 15, "unknown": 16, "interior_door": 17}
        self.room_map = {code: name for name, code in original_map.items()}
        self.round_value = round_value

    def _map_room_type(self, code: int) -> str:
        return self.room_map.get(code, str(code))

    def _segments_to_polygon(self, segments):
        lines = [LineString([(seg[0], seg[1]), (seg[2], seg[3])]) for seg in segments]
        
        polys = list(polygonize(lines))
        
        if len(polys) == 0:
            raise ValueError("No polygon could be formed from the provided segments.")
        elif len(polys) > 1:
            raise ValueError(f"More than one polygon was detected: {len(polys)} polygons formed.")
        
        return polys[0]

    def _convert_entry(self, data: Dict[str, Any]) -> Dict[str, Any]:
        rooms = []
        total_area = 0.0

        # for room_idx in range(len(data["room_type"])):
        #     room = {
        #         "id": room_idx,              
        #         "room_type": self._map_room_type(data["room_type"][room_idx]),
        #         "segments": [],
        #         "floor_polygon": []
        #     }
        #     rooms.append(room)

        # for edge_idx, edge in enumerate(data["edges"]):
        #     associated_room_indices = data["ed_rm"][edge_idx]
        #     polygone = []
        #     for room_idx in associated_room_indices:
        #         coords = edge[:4]
        #         rooms[room_idx]["segments"].append(coords)

        # for room in rooms:
        #     if len(room["segments"]) > 0:
        #         poly = self._segments_to_polygon(room["segments"])
        #         area = round(poly.area, self.round_value)
        #         minx, miny, maxx, maxy = poly.bounds
        #         width  = round(maxx - minx, self.round_value)
        #         height = round(maxy - miny, self.round_value)
        #         is_rectangular = 1 if len(poly.exterior.coords) - 1 == 4 else 0
        #         floor_polygon = [
        #             {'x': round(x, self.round_value), 'y': round(y, self.round_value)}
        #             for x, y in poly.exterior.coords[:-1]
        #         ]

        #         total_area += area
        #         del room["segments"]

        #         room["area"] = area
        #         room["width"] = width
        #         room["height"] = height
        #         room["is_regular"] = is_rectangular
        #         room["floor_polygon"] = floor_polygon

        rooms = [{
            "id": i,
            "room_type": self._map_room_type(rt),
            "segments": [] 
        } for i, rt in enumerate(data["room_type"])]

        for edge, room_indices in zip(data["edges"], data["ed_rm"]):
            segment = edge[:4] 
            for room_idx in room_indices:
                rooms[room_idx]["segments"].append(segment)

        for room in rooms:
            segments = room.pop("segments", [])
            if segments:
                poly = self._segments_to_polygon(segments)
                area = round(poly.area, self.round_value)
                minx, miny, maxx, maxy = poly.bounds
                room.update({
                    "area": area,
                    "width": round(maxx - minx, self.round_value),
                    "height": round(maxy - miny, self.round_value),
                    "is_regular": 1 if len(poly.exterior.coords) - 1 == 4 else 0,
                    "floor_polygon": [
                        {'x': round(x, self.round_value), 'y': round(y, self.round_value)}
                        for x, y in poly.exterior.coords[:-1]
                    ]
                })
                total_area += area

        return {
            'room_count': len(rooms),
            'total_area': round(total_area, self.round_value),
            'room_types': [r['room_type'] for r in rooms],
            'rooms': rooms,
            # 'edges': edges
        }

    def create_dataset(self, raw_data: List[Dict[str, Any]]) -> DatasetDict:
        # train_split, temp_split = train_test_split(raw_data, test_size=0.2, shuffle=False)
        # test_split, validation_split = train_test_split(temp_split, test_size=0.5, shuffle=False)
        train_split, test_split = train_test_split(raw_data, test_size=0.5, shuffle=False)

        train_list=[self._convert_entry(i) for i in train_split]
        test_list=[self._convert_entry(i) for i in test_split]
        # validation_list=[self._convert_entry(i) for i in validation_split]

        return DatasetDict({
            'train': Dataset.from_list(train_list),
            'test': Dataset.from_list(test_list),
            # 'validation': Dataset.from_list(validation_list)
        })

    def _load_folder(self, folder: str) -> List[Dict[str, Any]]:
        files = sorted(Path(folder).glob("*.json"), key=lambda p: int(p.stem))
        return [json.loads(p.read_text()) for p in files]

    def __call__(self, raw_data: List[Dict[str, Any]]) -> DatasetDict:
        rplan_json = self._load_folder(raw_data)
        return self.create_dataset(rplan_json)

if __name__ == "__main__":
    rplan_json = "datasets/rplan_json_test"
    converter = RPLANConverter()
    dataset = converter(rplan_json) 
    print(dataset['train'][0])
