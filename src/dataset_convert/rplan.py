from typing import List, Dict, Any
from pathlib import Path
import json
from datasets import DatasetDict, Dataset
from sklearn.model_selection import train_test_split
from shapely import Polygon
from shapely.geometry import LineString
from shapely.ops import polygonize
from shapely.affinity import scale
from tqdm import tqdm

class RPLANConverter:
    def __init__(self, round_value: int = 1):
        original_map = {"living_room": 1, "kitchen": 2, "bedroom": 3, "bathroom": 4, "balcony": 5, "entrance": 6, "dining_room": 7, "study_room": 8, "storage": 10 , "front_door": 15, "unknown": 16, "interior_door": 17}
        self.room_map = {code: name for name, code in original_map.items()}
        self.round_value = round_value
        # Each floor plan is represented as vector graphics in an 18m x 18m square and converted into a 256x256 image 
        self.pixel_to_meter = 18 / 256

    def _map_room_type(self, code: int) -> str:
        return self.room_map.get(code, str(code))
    
    def _segments_to_polygon(self, segments):
        segments = [list(seg) for seg in {tuple(seg) for seg in segments}]
        lines = [LineString([(seg[0], seg[1]), (seg[2], seg[3])]) for seg in segments]
        
        polys = list(polygonize(lines))

        if len(polys) == 0:
            raise ValueError("No polygon could be formed from the provided segments.")
        elif len(polys) > 1:
            for i, poly in enumerate(polys, start=1):
                print(f"Polygon {i}:\n{poly.wkt}\n")
            raise ValueError(f"More than one polygon was detected: {len(polys)} polygons formed.")
        
        return polys[0]

    def _convert_entry(self, data: Dict[str, Any]) -> Dict[str, Any]:
        rooms = []
        total_area = 0.0

        rooms = [{
            "id": i,
            "room_type": self._map_room_type(rt),
            "segments": [] 
        } for i, rt in enumerate(data["room_type"])]

        for edge, room_indices in zip(data["edges"], data["ed_rm"]):
            segment = edge[:4] 
            for room_idx in room_indices[:1]:
                rooms[room_idx]["segments"].append(segment)

        for room in rooms:
            segments = room.pop("segments", [])
            if segments:
                try:
                    poly = self._segments_to_polygon(segments)
                    poly = scale(poly, xfact=self.pixel_to_meter, yfact=self.pixel_to_meter, origin=(0, 0))
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
            
                except Exception as e:
                    print(f"Error processing segments for room {room['id']}: {e}")
                    return None
            else:
                return None

        return {
            'room_count': len(rooms),
            'total_area': round(total_area, self.round_value),
            'room_types': [r['room_type'] for r in rooms],
            'rooms': rooms,
            'edges': data["ed_rm"],
        }

    def create_dataset(self, raw_data: List[Dict[str, Any]]) -> DatasetDict:
        train_data, temp_data = train_test_split(raw_data, test_size=0.2, shuffle=False)
        test_data, validation_data = train_test_split(temp_data, test_size=0.5, shuffle=False)

        train_list = [
            converted
            for i in tqdm(train_data, desc='Processing training data')
            if (converted := self._convert_entry(i)) is not None
        ]
        test_list = [
            converted
            for i in tqdm(test_data, desc='Processing test data')
            if (converted := self._convert_entry(i)) is not None
        ]
        validation_list = [
            converted
            for i in tqdm(validation_data, desc='Processing validation data')
            if (converted := self._convert_entry(i)) is not None
        ]
        return DatasetDict({
            'train': Dataset.from_list(train_list),
            'test': Dataset.from_list(test_list),
            'validation': Dataset.from_list(validation_list)
        })

    def _load_folder(self, folder: str) -> List[Dict[str, Any]]:
        files = sorted(Path(folder).glob("*.json"), key=lambda p: int(p.stem))
        return [json.loads(p.read_text()) for p in tqdm(files, desc="Loading JSON files")]

    def __call__(self, raw_data: List[Dict[str, Any]]) -> DatasetDict:
        rplan_json = self._load_folder(raw_data)
        return self.create_dataset(rplan_json)

# if __name__ == "__main__":
#     rplan_json = "datasets/rplan_json"
#     converter = RPLANConverter()
#     dataset = converter(rplan_json) 
#     dataset.save_to_disk("datasets/rplan_converted")
#     print("Dataset saved to disk.")
#     print(dataset)
#     print(dataset["validation"][0])

