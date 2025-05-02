from src.pred.extract_output_json import extract_output_json
from src.pred.feedback_generator import FeedbackGenerator
from src.utils.create_example import create_input
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest
from datasets import load_from_disk

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
    system_prompt = (
        f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n {SYSTEM_PROMPT} \n"
    )
    prompt = (
        f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n"
        f"{system_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>\n"
        f"{create_input(sample)}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n"
    )
    return prompt

def select_least_overlap(candidates, input_prompt):
    return min(
        candidates,
        key=lambda cand: FeedbackGenerator.analyze(
            extract_output_json(cand.text),
            input_prompt
        )['total_overlap_area']
    )

llm = LLM(
    model="models/Llama-3.3-70B-Instruct",
    tensor_parallel_size=4,
    device="cuda",
    enable_lora=True
)
print("Model loaded")
lora_request = LoRARequest("floorplan_adapter", 1, "output/rplan_25_70B/")

sampling_params = SamplingParams(
    temperature=0.7,
    top_p=0.9,
    max_tokens=4096,
    n=50,
    best_of=50
)

dataset = load_from_disk("datasets/rplan_converted")["test"]
prompts = [
    build_prompt(dataset[3])
]

outputs = llm.generate(
    prompts,
    sampling_params,
    lora_request=lora_request
)

res = select_least_overlap(outputs[0].outputs, create_input(dataset[3], is_str=False))
print(res.text)

