import random
from datasets import load_from_disk
from src.utils.create_example import create_input, create_output

# def create_input(sample, is_str=True):
#     inp = {
#         "room_count": sample.get("room_count"),
#         "total_area": sample.get("total_area"),
#         "room_types": sample.get("room_types"),
#         "rooms": [
#             {
#                 "room_type": room.get("room_type"),
#                 "width": room.get("width"),
#                 "height": room.get("height"),
#                 "is_regular": room.get("is_regular"),
#             }
#             for room in sample.get("rooms", [])
#         ],
#         # "edges": sample.get("edges", []),
#     }
#     if is_str:
#         return str({"input": inp})
#     else:
#         return inp
    
# def create_output(sample):
#     if isinstance(sample, dict) and "edges" in sample:
#         sample.pop("edges")
#     output = {"output": sample}
#     return str(output)
    
def get_few_shot_examples(example):
    return f"{create_input(example)}\n   {create_output(example)}"

if __name__ == "__main__":
    dataset = load_from_disk("datasets/rplan_converted_no_doors")
    print(dataset)
    train_examples = list((dataset["train"]))
    random_examples = random.sample(train_examples, 10)

    for idx, example in enumerate(random_examples):
        few_shot_examples = get_few_shot_examples(example)
        print(f"{idx+1}. {few_shot_examples}\n")