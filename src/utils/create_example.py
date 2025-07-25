import json
from src.utils.constants import SYSTEM_PROMPT
    
def create_output(sample):
    output = {
        "room_count": sample.get("room_count"),
        "total_area": sample.get("total_area"),
        "spaces": [
            {
                "id": room.get("id"),
                "room_type": room.get("room_type"),
                "area": room.get("area"),
                "floor_polygon": room.get("floor_polygon"),
            }
            for room in sample.get("spaces", [])
        ],
    }
    return json.dumps({"output": output})

def build_prompt(sample):
    prompt = (
        f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n"
        f"{SYSTEM_PROMPT}<|eot_id|><|start_header_id|>user<|end_header_id|>\n"
        f"{sample.get('prompt', '{}')}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n"
    )
    return prompt
