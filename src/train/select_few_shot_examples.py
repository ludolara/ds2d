import random
from datasets import load_from_disk
import json
    
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
    
def get_few_shot_examples(example):
    return f"{example['prompt']}\n   {create_output(example)}"

if __name__ == "__main__":
    dataset = load_from_disk("../../datasets/final/rplan_5")
    print(dataset)
    train_examples = list((dataset["train"]))
    random_examples = random.sample(train_examples, 10)

    for idx, example in enumerate(random_examples):
        few_shot_examples = get_few_shot_examples(example)
        # print(f"{idx+1}. {few_shot_examples}\n")