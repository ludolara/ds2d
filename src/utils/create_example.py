import json
from src.utils.constants import SYSTEM_PROMPT

def create_input(sample, is_str=True):
    inp = {
        "room_count": sample.get("room_count"),
        "total_area": sample.get("total_area"),
        "rooms": [
            {
                "id": room.get("id"),
                "room_type": room.get("room_type"),
                "width": room.get("width"),
                "height": room.get("height"),
                "is_rectangular": room.get("is_rectangular"),
            }
            for room in sample.get("rooms", [])
        ],
        "input_graph": json.loads(sample.get("input_graph", [])),
    }
    if is_str:
        return str({"input": inp})
    else:
        return inp
    
def create_output(sample):
    output = {
        "room_count": sample.get("room_count"),
        "rooms": [
            {
                "id": room.get("id"),
                "room_type": room.get("room_type"),
                "area": room.get("area"),
                "floor_polygon": room.get("floor_polygon"),
            }
            for room in sample.get("rooms", [])
        ],
    }
    return str({"output": output})

def build_prompt(sample):
    prompt = (
        f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n"
        f"{SYSTEM_PROMPT}<|eot_id|><|start_header_id|>user<|end_header_id|>\n"
        f"{sample}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n"
    )
    return prompt
