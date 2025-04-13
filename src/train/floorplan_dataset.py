from datasets import load_from_disk, concatenate_datasets

def create_input(sample):
    inp = {
        "input": {
            'room_count': sample.get("room_count"),
            'total_area': sample.get("total_area"),
            'room_types': sample.get("room_types"),
            'rooms': [
                {
                    "room_type": room.get("room_type"),
                    "width": room.get("width"),
                    "height": room.get("height"),
                    **({"is_regular": room["is_regular"]} if "is_regular" in room else {})
                }
                for room in sample.get("rooms", [])
            ],
            "edges": sample.get("edges", [])
        }
    }
    return str(inp)

def create_output(sample):
    if isinstance(sample, dict) and "edges" in sample:
        sample.pop("edges")
    output = {"output": sample}
    return str(output)

def get_custom_dataset(dataset_config, tokenizer, split):
    rplan = load_from_disk('datasets/rplan_converted')[split]
    # procthor = load_from_disk('datasets/DStruct2Design-nightly')[split]
    # dataset = concatenate_datasets([rplan, procthor])

    dataset = concatenate_datasets([rplan])
    dataset = dataset.shuffle(seed=42)

    def process_sample(sample):
        system_prompt = "you are to generate a floor plan in a JSON structure. you have to satisfy the adjacency constraints given as pairs of neighboring rooms; two connecting rooms are presented as (room_type1 room_id1, room_type2 room_id2). you also need to satisfy additional contraints given by the user."

        prompt = (
            f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n"
            f"{system_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>\n"
            f"{create_input(sample)}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n"
        )
        
        prompt_tokens = tokenizer(prompt, add_special_tokens=False)
        
        response = f"{create_output(sample)}<|eot_id|>"
        response_tokens = tokenizer(response, add_special_tokens=False)
        
        input_ids = prompt_tokens['input_ids'] + response_tokens['input_ids']
        attention_mask = [1] * len(input_ids)
        
        labels = [-100] * len(prompt_tokens['input_ids']) + response_tokens['input_ids']
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels
        }
    
    return dataset.map(process_sample, remove_columns=dataset.column_names)
