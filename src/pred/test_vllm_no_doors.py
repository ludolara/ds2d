from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

def build_prompt(sample):
    SYSTEM_PROMPT = """
You are a state-of-the-art floor-plan generator that translates JSON specifications and connectivity requirements defined by a bubble diagram into precise, optimized layouts. 
Your algorithm considers each room's dimensions, proportion, and desired adjacencies to produce an efficient arrangement that maximizes usable space while honoring all constraints.
Your top priority is that no two room polygons ever overlap. Rooms must be strictly disjoint, doors may touch room boundaries, but room interiors must never intersect.  

Your output must be a JSON object, where `output` key contains:
- `room_count`: the total number of room and door entries  
- `rooms`: a list of mixing rooms and doors. Each room or door entry in `room` must include:
 - `id`: formatted as `<room_type>|<unique_index>` (e.g. `"bedroom|2"` or `"interior_door|0"`).
 - `room_type`: the room type (e.g. `"living_room"`, `"kitchen"`, etc.)
 - `area` in square meters (all positive numbers)  
 - `floor_polygon`: an ordered list of `{x: , y:}` vertices defining a simple polygon, minimun of 4 vertices, and maximum of 20 vertices. The doors must connect two rooms.

Additional rules:
- **Absolute non-overlap**: no two room polygons may share any interior point under any circumstances.
- Every adjacency in the bubble diagram must be bridged by exactly one door.  
- Every `id` used in the bubble diagram and on any door must appear in the `rooms` list.  

These are 3 examples of the expected output:

1. {'input': {'room_count': 6, 'total_area': 62.9, 'room_types': ['bedroom', 'bathroom', 'kitchen', 'bedroom', 'living_room', 'bedroom'], 'rooms': [{'id': 'bedroom|0', 'room_type': 'bedroom', 'width': 3.1, 'height': 3.4, 'is_rectangular': 1}, {'id': 'bathroom', 'room_type': 'bathroom', 'width': 2.0, 'height': 1.8, 'is_rectangular': 1}, {'id': 'kitchen', 'room_type': 'kitchen', 'width': 2.5, 'height': 1.7, 'is_rectangular': 1}, {'id': 'bedroom|1', 'room_type': 'bedroom', 'width': 2.5, 'height': 3.2, 'is_rectangular': 1}, {'id': 'living_room', 'room_type': 'living_room', 'width': 5.0, 'height': 9.1, 'is_rectangular': 0}, {'id': 'bedroom|2', 'room_type': 'bedroom', 'width': 2.5, 'height': 4.1, 'is_rectangular': 0}], 'bubble_diagram': {'bedroom|0': ['living_room'], 'bathroom': ['living_room'], 'kitchen': ['living_room'], 'bedroom|1': ['living_room'], 'living_room': ['bedroom|0', 'bathroom', 'kitchen', 'bedroom|1', 'bedroom|2'], 'bedroom|2': ['living_room']}}}
   {'output': {'room_count': 6, 'rooms': [{'id': 'bedroom|0', 'room_type': 'bedroom', 'area': 10.7, 'floor_polygon': [{'x': 9.6, 'y': 15.1}, {'x': 12.7, 'y': 15.1}, {'x': 12.7, 'y': 11.7}, {'x': 9.6, 'y': 11.7}]}, {'id': 'bathroom', 'room_type': 'bathroom', 'area': 3.7, 'floor_polygon': [{'x': 10.6, 'y': 9.4}, {'x': 10.6, 'y': 11.2}, {'x': 12.7, 'y': 11.2}, {'x': 12.7, 'y': 9.4}]}, {'id': 'kitchen', 'room_type': 'kitchen', 'area': 4.3, 'floor_polygon': [{'x': 12.1, 'y': 9.0}, {'x': 12.1, 'y': 7.3}, {'x': 9.6, 'y': 7.3}, {'x': 9.6, 'y': 9.0}]}, {'id': 'bedroom|1', 'room_type': 'bedroom', 'area': 8.0, 'floor_polygon': [{'x': 12.1, 'y': 6.9}, {'x': 12.1, 'y': 3.7}, {'x': 9.6, 'y': 3.7}, {'x': 9.6, 'y': 6.9}]}, {'id': 'living_room', 'room_type': 'living_room', 'area': 27.2, 'floor_polygon': [{'x': 7.2, 'y': 7.2}, {'x': 7.2, 'y': 7.3}, {'x': 6.7, 'y': 7.3}, {'x': 6.7, 'y': 11.7}, {'x': 5.3, 'y': 11.7}, {'x': 5.3, 'y': 15.1}, {'x': 9.1, 'y': 15.1}, {'x': 9.1, 'y': 11.2}, {'x': 10.3, 'y': 11.2}, {'x': 10.3, 'y': 9.4}, {'x': 9.1, 'y': 9.4}, {'x': 9.1, 'y': 6.0}, {'x': 8.3, 'y': 6.0}, {'x': 8.3, 'y': 7.2}]}, {'id': 'bedroom|2', 'room_type': 'bedroom', 'area': 8.0, 'floor_polygon': [{'x': 7.2, 'y': 7.0}, {'x': 8.0, 'y': 7.0}, {'x': 8.0, 'y': 5.7}, {'x': 9.1, 'y': 5.7}, {'x': 9.1, 'y': 2.9}, {'x': 6.7, 'y': 2.9}, {'x': 6.7, 'y': 5.6}, {'x': 7.2, 'y': 5.6}]}]}}

2. {'input': {'room_count': 8, 'total_area': 93.2, 'room_types': ['bedroom', 'bedroom', 'bathroom', 'bedroom', 'balcony', 'living_room', 'kitchen', 'bathroom'], 'rooms': [{'id': 'bedroom|0', 'room_type': 'bedroom', 'width': 2.5, 'height': 4.4, 'is_rectangular': 0}, {'id': 'bedroom|1', 'room_type': 'bedroom', 'width': 3.0, 'height': 4.3, 'is_rectangular': 1}, {'id': 'bathroom|0', 'room_type': 'bathroom', 'width': 2.2, 'height': 1.4, 'is_rectangular': 1}, {'id': 'bedroom|2', 'room_type': 'bedroom', 'width': 3.0, 'height': 3.7, 'is_rectangular': 1}, {'id': 'balcony', 'room_type': 'balcony', 'width': 6.5, 'height': 1.4, 'is_rectangular': 0}, {'id': 'living_room', 'room_type': 'living_room', 'width': 6.8, 'height': 8.5, 'is_rectangular': 0}, {'id': 'kitchen', 'room_type': 'kitchen', 'width': 3.2, 'height': 1.5, 'is_rectangular': 1}, {'id': 'bathroom|1', 'room_type': 'bathroom', 'width': 1.8, 'height': 3.5, 'is_rectangular': 1}], 'bubble_diagram': {'bedroom|0': ['living_room'], 'bedroom|1': ['living_room', 'bathroom|0'], 'bathroom|0': ['bedroom|1'], 'bedroom|2': ['living_room'], 'balcony': ['living_room'], 'living_room': ['bedroom|0', 'bedroom|1', 'bedroom|2', 'balcony', 'bathroom|1', 'kitchen'], 'kitchen': ['living_room'], 'bathroom|1': ['living_room']}}}
   {'output': {'room_count': 8, 'rooms': [{'id': 'bedroom|0', 'room_type': 'bedroom', 'area': 10.3, 'floor_polygon': [{'x': 4.9, 'y': 8.8}, {'x': 4.9, 'y': 9.4}, {'x': 3.7, 'y': 9.4}, {'x': 3.7, 'y': 13.2}, {'x': 6.2, 'y': 13.2}, {'x': 6.2, 'y': 8.8}]}, {'id': 'bedroom|1', 'room_type': 'bedroom', 'area': 13.0, 'floor_polygon': [{'x': 13.7, 'y': 8.9}, {'x': 10.7, 'y': 8.9}, {'x': 10.7, 'y': 13.2}, {'x': 13.7, 'y': 13.2}]}, {'id': 'bathroom|0', 'room_type': 'bathroom', 'area': 3.2, 'floor_polygon': [{'x': 14.3, 'y': 7.1}, {'x': 12.0, 'y': 7.1}, {'x': 12.0, 'y': 8.5}, {'x': 14.3, 'y': 8.5}]}, {'id': 'bedroom|2', 'room_type': 'bedroom', 'area': 11.3, 'floor_polygon': [{'x': 13.7, 'y': 6.7}, {'x': 13.7, 'y': 3.0}, {'x': 10.7, 'y': 3.0}, {'x': 10.7, 'y': 6.7}]}, {'id': 'balcony', 'room_type': 'balcony', 'area': 8.8, 'floor_polygon': [{'x': 3.7, 'y': 13.6}, {'x': 3.7, 'y': 14.2}, {'x': 4.1, 'y': 14.2}, {'x': 4.1, 'y': 15.0}, {'x': 10.3, 'y': 15.0}, {'x': 10.3, 'y': 13.6}]}, {'id': 'living_room', 'room_type': 'living_room', 'area': 33.4, 'floor_polygon': [{'x': 7.0, 'y': 7.5}, {'x': 4.9, 'y': 7.5}, {'x': 4.9, 'y': 8.4}, {'x': 6.6, 'y': 8.4}, {'x': 6.6, 'y': 13.2}, {'x': 10.3, 'y': 13.2}, {'x': 10.3, 'y': 8.5}, {'x': 11.6, 'y': 8.5}, {'x': 11.6, 'y': 7.1}, {'x': 10.3, 'y': 7.1}, {'x': 10.3, 'y': 4.7}, {'x': 7.0, 'y': 4.7}]}, {'id': 'kitchen', 'room_type': 'kitchen', 'area': 4.8, 'floor_polygon': [{'x': 7.0, 'y': 3.0}, {'x': 7.0, 'y': 4.4}, {'x': 10.3, 'y': 4.4}, {'x': 10.3, 'y': 3.0}]}, {'id': 'bathroom|1', 'room_type': 'bathroom', 'area': 6.2, 'floor_polygon': [{'x': 6.6, 'y': 7.0}, {'x': 6.6, 'y': 3.5}, {'x': 4.9, 'y': 3.5}, {'x': 4.9, 'y': 7.0}]}]}}

3. {'input': {'room_count': 6, 'total_area': 68.5, 'room_types': ['balcony', 'bedroom', 'kitchen', 'bedroom', 'bathroom', 'living_room'], 'rooms': [{'id': 'balcony', 'room_type': 'balcony', 'width': 3.0, 'height': 1.1, 'is_rectangular': 1}, {'id': 'bedroom|0', 'room_type': 'bedroom', 'width': 2.7, 'height': 4.1, 'is_rectangular': 1}, {'id': 'kitchen', 'room_type': 'kitchen', 'width': 2.7, 'height': 1.5, 'is_rectangular': 1}, {'id': 'bedroom|1', 'room_type': 'bedroom', 'width': 2.7, 'height': 4.2, 'is_rectangular': 1}, {'id': 'bathroom', 'room_type': 'bathroom', 'width': 1.7, 'height': 1.7, 'is_rectangular': 1}, {'id': 'living_room', 'room_type': 'living_room', 'width': 5.0, 'height': 9.0, 'is_rectangular': 0}], 'bubble_diagram': {'balcony': ['living_room'], 'bedroom|0': ['living_room'], 'kitchen': ['living_room'], 'bedroom|1': ['living_room'], 'bathroom': ['living_room'], 'living_room': ['balcony', 'bedroom|0', 'kitchen', 'bedroom|1', 'bathroom']}}}
   {'output': {'room_count': 6, 'rooms': [{'id': 'balcony', 'room_type': 'balcony', 'area': 3.3, 'floor_polygon': [{'x': 9.5, 'y': 14.9}, {'x': 12.4, 'y': 14.9}, {'x': 12.4, 'y': 13.8}, {'x': 9.5, 'y': 13.8}]}, {'id': 'bedroom|0', 'room_type': 'bedroom', 'area': 11.2, 'floor_polygon': [{'x': 9.2, 'y': 9.4}, {'x': 6.5, 'y': 9.4}, {'x': 6.5, 'y': 13.5}, {'x': 9.2, 'y': 13.5}]}, {'id': 'kitchen', 'room_type': 'kitchen', 'area': 4.2, 'floor_polygon': [{'x': 9.2, 'y': 9.1}, {'x': 9.2, 'y': 7.6}, {'x': 6.5, 'y': 7.6}, {'x': 6.5, 'y': 9.1}]}, {'id': 'bedroom|1', 'room_type': 'bedroom', 'area': 11.6, 'floor_polygon': [{'x': 5.0, 'y': 7.3}, {'x': 7.7, 'y': 7.3}, {'x': 7.7, 'y': 3.1}, {'x': 5.0, 'y': 3.1}]}, {'id': 'bathroom', 'room_type': 'bathroom', 'area': 2.8, 'floor_polygon': [{'x': 8.0, 'y': 3.1}, {'x': 8.0, 'y': 4.8}, {'x': 9.7, 'y': 4.8}, {'x': 9.7, 'y': 3.1}]}, {'id': 'living_room', 'room_type': 'living_room', 'area': 34.7, 'floor_polygon': [{'x': 10.0, 'y': 5.1}, {'x': 8.0, 'y': 5.1}, {'x': 8.0, 'y': 7.3}, {'x': 9.5, 'y': 7.3}, {'x': 9.5, 'y': 13.5}, {'x': 13.0, 'y': 13.5}, {'x': 13.0, 'y': 4.5}, {'x': 10.0, 'y': 4.5}]}]}}

   Return only a JSON object containing an `output` key without extra commentary or explanation. Veriify with code if it has overlaps if yes try again.
"""

    system_prompt = (
        f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n {SYSTEM_PROMPT} \n"
    )
    prompt = (
        f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n"
        f"{system_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>\n"
        f"{sample}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n"
    )
    return prompt

model_path = "models/Llama-4-Scout-17B-16E-Instruct"
# lora_adapter_path = "/home/l/luislara/links/scratch/ds2d_v2/output/rplan_3_70B/"

model = LLM(model=model_path, tensor_parallel_size=4, device="cuda", max_model_len=100_000, 
            override_generation_config= {
                "attn_temperature_tuning": True,
            }
            # enable_lora=True
        )
# lora = LoRARequest("lora_adapter", 1, lora_adapter_path)
sampling_params = SamplingParams(max_tokens=4096)

sample = """
{'input': {'room_count': 5, 'total_area': 64.4, 'room_types': ['bedroom', 'living_room', 'bedroom', 'kitchen', 'bathroom'], 'rooms': [{'id': 'bedroom|0', 'room_type': 'bedroom', 'width': 3.0, 'height': 4.9, 'is_rectangular': 1}, {'id': 'living_room', 'room_type': 'living_room', 'width': 5.9, 'height': 5.1, 'is_rectangular': 0}, {'id': 'bedroom|1', 'room_type': 'bedroom', 'width': 2.7, 'height': 3.7, 'is_rectangular': 1}, {'id': 'kitchen', 'room_type': 'kitchen', 'width': 2.7, 'height': 2.7, 'is_rectangular': 1}, {'id': 'bathroom', 'room_type': 'bathroom', 'width': 1.7, 'height': 3.3, 'is_rectangular': 1}], 'bubble_diagram': {'bedroom|0': ['living_room'], 'living_room': ['bedroom|0', 'bathroom', 'bedroom|1', 'kitchen'], 'bedroom|1': ['living_room'], 'kitchen': ['living_room'], 'bathroom': ['living_room']}}}
"""

outputs = model.generate(
    [build_prompt(sample)], 
    sampling_params, 
    # lora_request=lora
)
print(outputs[0].outputs[0].text)
