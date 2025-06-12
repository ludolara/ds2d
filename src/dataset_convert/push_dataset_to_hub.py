import os
from dotenv import load_dotenv 
from huggingface_hub import HfApi, login
from datasets import load_from_disk
# from src.utils.create_example import create_input
import json

def create_input(sample, is_str=True):
    inp = {
        "room_count": sample.get("room_count"),
        "total_area": sample.get("total_area"),
        # "room_types": sample.get("room_types"),
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
    
# load_dotenv()  
# login(token=os.environ.get("HF_TOKEN_WRITE"))
api = HfApi()
api.create_repo(repo_id="ludolara/rplan", repo_type="dataset", private=False, exist_ok=True)
rplan = load_from_disk('datasets/rplan')
rplan = rplan.map(lambda sample: {"input": create_input(sample)})
rplan.push_to_hub("ludolara/rplan")

