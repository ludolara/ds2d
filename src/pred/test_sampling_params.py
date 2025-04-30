from src.pred.extract_output_json import extract_output_json
from src.pred.feedback_generator import FeedbackGenerator
from src.utils.constants import SYSTEM_PROMPT
from src.utils.create_example import create_input
from vllm import LLM, SamplingParams
import json
from shapely.geometry import Polygon
from vllm.lora.request import LoRARequest
from datasets import load_from_disk
from transformers import AutoTokenizer

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

class PolygonValidator():
    def __call__(self, input_ids, scores, logits):
        if input_ids[0, -1] != tokenizer.eos_token_id and chr(input_ids[0, -1].item()) != '}':
            return logits

        txt = tokenizer.decode(input_ids[0].cpu().numpy())
        try:
            data = json.loads(txt.replace("'", '"'))["output"]
            polys = [Polygon([(p["x"], p["y"]) for p in room["floor_polygon"]])
                     for room in data["rooms"]]
            for i in range(len(polys)):
                for j in range(i+1, len(polys)):
                    if polys[i].intersects(polys[j]) and not polys[i].touches(polys[j]):
                        logits[:] = -float("inf")
                        logits[tokenizer.eos_token_id] = 0.0
                        return logits
        except Exception:
            pass  
        return logits

# def polygon_validator(token_ids, logits):
#     try:
#         partial_text = tokenizer.decode(token_ids)  # You'd need the model's tokenizer
#         if is_valid_floor_plan(partial_text):
#             # If valid, return logits unchanged
#             return logits
#         else:
#             # If invalid (overlapping rooms detected), mask out all tokens except special ones
#             # This effectively prevents the model from continuing with invalid configurations
#             logits[:] = -9999.999
#             # Optionally allow EOS token to be generated
#             logits[tokenizer.eos_token_id] = 0.0
#             return logits
#     except:
#         # If we can't parse or validate yet, let generation continue
#         return logits

# def is_valid_floor_plan(partial_text):
#     try:
#         # data = json.loads(partial_text + "}") if not partial_text.endswith("}") else json.loads(partial_text)
#         json_str = partial_text.replace("'", '"')
#         data = json.loads(json_str)
#         data = data["output"]

#         rooms = []
#         for item in data.get("rooms", []):
#             if "floor_polygon" in item and len(item["floor_polygon"]) >= 3:  
#                 rooms.append(item)
        
#         for i, room1 in enumerate(rooms):
#             poly1 = Polygon(room1["floor_polygon"])
#             for j, room2 in enumerate(rooms):
#                 if i >= j:  
#                     continue
                    
#                 poly2 = Polygon(room2["floor_polygon"])
#                 if poly1.intersects(poly2) and not poly1.touches(poly2):
#                     return False
                    
#         return True
#     except (json.JSONDecodeError, KeyError, TypeError):
#         return True

def select_least_overlap(candidates, input_prompt):
    return min(
        candidates,
        key=lambda cand: FeedbackGenerator.analyze(
            # json.loads(cand.text.replace("'", '"'))['output'],
            extract_output_json(cand.text),
            input_prompt
        )['total_overlap_area']
    )

# tokenizer = AutoTokenizer.from_pretrained("models/Llama-3.3-70B-Instruct")
# def process_token(token_ids, logits):
#     partial_text = tokenizer.decode(token_ids)
#     print(partial_text)
#     return logits

# Usage with vLLM
# llm = LLM(model="your-floor-plan-model")
llm = LLM(
    model="models/Llama-3.3-70B-Instruct",
    tensor_parallel_size=4,
    device="cuda",
    enable_lora=True
)
print("Model loaded")
lora_request = LoRARequest("floorplan_adapter", 1, "output/rplan_25_70B/")

# Configure sampling with logits processor
sampling_params = SamplingParams(
    temperature=0.7,
    top_p=0.9,
    max_tokens=4096,
    n=50,
    best_of=50,
    # presence_penalty=0.5,
    # logits_processors=[process_token]
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
print(res)

# print(prompts[0])
# print(outputs[0].outputs[0].text)

for idx, completion in enumerate(outputs[0].outputs):
    try:
        json_str = completion.text.replace("'", '"')
        data = json.loads(json_str)
        output = data["output"]

        input_prompt = create_input(dataset[3], is_str=False)
        stats = FeedbackGenerator.analyze(output, input_prompt)
        print(idx)
        print(stats["total_overlap_area"])
    except Exception as e:
        print("Error:", e)
        continue
