from vllm import LLM, SamplingParams

SYSTEM_PROMPT = """
You are a state-of-the-art floor-plan generator that translates JSON specifications and connectivity requirements defined by a bubble diagram into precise, optimized layouts. 
Your algorithm considers each room's dimensions, proportion, and desired adjacencies to produce an efficient arrangement that maximizes usable space while honoring all constraints.
Your top priority is that no two room polygons ever overlap. Rooms must be strictly disjoint, doors may touch room boundaries, but room interiors must never intersect.  

Your output must be a JSON object, where `output` key contains:
- `room_count`: the total number of room and door entries  
- `rooms`: a list of mixing rooms and doors. Each room or door entry in `room` must include:
 - `id`: formatted as `<room_type>|<unique_index>` (e.g. `"bedroom|2"` or `"interior_door|0"`)  
 - `room_type`: the room type (e.g. `"living_room"`, `"kitchen"`, etc.)
 - `area` in square meters (all positive numbers)  
 - `floor_polygon`: an ordered list of `{x: , y:}` vertices defining a simple polygon  

Additional rules:
- **Absolute non-overlap**: no two room polygons may share any interior point under any circumstances.
- Every adjacency in the bubble diagram must be bridged by exactly one door.  
- Every `id` used in the bubble diagram and on any door must appear in the `rooms` list.  

Return only a JSON object containing an `output` key without extra commentary or explanation.
"""

def build_prompt(sample):
    prompt = (
        f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n"
        f"{SYSTEM_PROMPT}<|eot_id|><|start_header_id|>user<|end_header_id|>\n"
        f"{sample}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n"
    )
    return prompt

model = LLM(model="output/OLD_test-GRPO_3.3/checkpoint-1300")
sampling_params = SamplingParams(temperature=0.7, top_p=0.9, max_tokens=4096)

sample = """
{'input': {'room_count': 10, 'total_area': 72.4, 'rooms': [{'id': 'living_room', 'room_type': 'living_room', 'width': 7.0, 'height': 5.2, 'is_rectangular': 0}, {'id': 'bedroom|0', 'room_type': 'bedroom', 'width': 3.6, 'height': 3.0, 'is_rectangular': 1}, {'id': 'bedroom|1', 'room_type': 'bedroom', 'width': 3.2, 'height': 4.1, 'is_rectangular': 1}, {'id': 'bathroom', 'room_type': 'bathroom', 'width': 2.2, 'height': 2.6, 'is_rectangular': 1}, {'id': 'kitchen', 'room_type': 'kitchen', 'width': 3.6, 'height': 2.6, 'is_rectangular': 1}, {'id': 'interior_door|0', 'room_type': 'interior_door', 'width': 1.5, 'height': 0.1, 'is_rectangular': 1}, {'id': 'interior_door|1', 'room_type': 'interior_door', 'width': 0.1, 'height': 0.6, 'is_rectangular': 1}, {'id': 'interior_door|2', 'room_type': 'interior_door', 'width': 1.5, 'height': 0.1, 'is_rectangular': 1}, {'id': 'interior_door|3', 'room_type': 'interior_door', 'width': 0.1, 'height': 0.8, 'is_rectangular': 1}, {'id': 'front_door', 'room_type': 'front_door', 'width': 0.1, 'height': 0.8, 'is_rectangular': 1}], 'bubble_diagram': {'living_room': ['bedroom|1', 'kitchen', 'bedroom|0'], 'bedroom|0': ['living_room'], 'bedroom|1': ['living_room'], 'bathroom': ['kitchen'], 'kitchen': ['living_room', 'bathroom']}}}
"""
outputs = model.generate(
    [build_prompt(sample)], 
    sampling_params, 
    # lora_request=lora
)
print(outputs[0].outputs[0].text)
