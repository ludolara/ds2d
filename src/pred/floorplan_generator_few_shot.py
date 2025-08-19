import ast
import os
import json
from tqdm import tqdm
from dotenv import load_dotenv
from src.dataset_convert.rplan_graph import RPLANGraph
from src.utils.constants import SYSTEM_PROMPT
from src.pred.feedback_generator import FeedbackGenerator
from src.pred.extract_output_json import extract_output_json
from datasets import load_from_disk
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

load_dotenv()
CACHE_DIR = os.environ.get("TRANSFORMERS_CACHE")

FEW_SHOT_EXAMPLES = """
The following are examples to better understand the task:

1. {"input": {"room_count": 7, "total_area": 71.0, "spaces": [{"id": "kitchen", "room_type": "kitchen", "height": 1.8, "width": 2.2}, {"id": "bedroom|0", "room_type": "bedroom", "height": 3.0, "width": 2.7}, {"id": "bathroom", "room_type": "bathroom", "height": 1.3, "width": 1.5}, {"id": "bedroom|1", "room_type": "bedroom", "height": 3.6, "width": 3.0}, {"id": "balcony", "room_type": "balcony", "height": 1.1, "width": 5.8}, {"id": "bedroom|2", "room_type": "bedroom", "height": 3.6, "width": 2.3}, {"id": "living_room", "room_type": "living_room", "area": 30.9}, {"id": "front_door", "room_type": "front_door", "height": 1.0, "width": 0.3}], "input_graph": {"kitchen": ["living_room"], "bedroom|0": ["living_room"], "bathroom": ["living_room"], "bedroom|1": ["living_room"], "balcony": ["bedroom|2"], "bedroom|2": ["balcony", "living_room"], "living_room": ["kitchen", "bedroom|0", "bathroom", "bedroom|1", "bedroom|2", "front_door"], "front_door": ["living_room"]}}}
   {"output": {"room_count": 7, "total_area": 71.0, "spaces": [{"id": "kitchen", "room_type": "kitchen", "area": 4.0, "floor_polygon": [{"x": 6.7, "y": 3.6}, {"x": 8.9, "y": 3.6}, {"x": 8.9, "y": 1.8}, {"x": 6.7, "y": 1.8}]}, {"id": "bedroom|0", "room_type": "bedroom", "area": 8.1, "floor_polygon": [{"x": 9.2, "y": 4.9}, {"x": 11.9, "y": 4.9}, {"x": 11.9, "y": 1.8}, {"x": 9.2, "y": 1.8}]}, {"id": "bathroom", "room_type": "bathroom", "area": 2.1, "floor_polygon": [{"x": 11.9, "y": 9.3}, {"x": 10.3, "y": 9.3}, {"x": 10.3, "y": 10.6}, {"x": 11.9, "y": 10.6}]}, {"id": "bedroom|1", "room_type": "bedroom", "area": 10.8, "floor_polygon": [{"x": 8.9, "y": 14.6}, {"x": 11.9, "y": 14.6}, {"x": 11.9, "y": 11.0}, {"x": 8.9, "y": 11.0}]}, {"id": "balcony", "room_type": "balcony", "area": 6.5, "floor_polygon": [{"x": 6.1, "y": 15.0}, {"x": 6.1, "y": 16.2}, {"x": 11.9, "y": 16.2}, {"x": 11.9, "y": 15.0}]}, {"id": "bedroom|2", "room_type": "bedroom", "area": 8.3, "floor_polygon": [{"x": 8.4, "y": 11.0}, {"x": 6.1, "y": 11.0}, {"x": 6.1, "y": 14.6}, {"x": 8.4, "y": 14.6}]}, {"id": "living_room", "room_type": "living_room", "area": 30.9, "floor_polygon": [{"x": 11.9, "y": 5.3}, {"x": 8.9, "y": 5.3}, {"x": 8.9, "y": 4.9}, {"x": 8.9, "y": 4.9}, {"x": 8.9, "y": 4.0}, {"x": 6.1, "y": 4.0}, {"x": 6.1, "y": 10.6}, {"x": 9.9, "y": 10.6}, {"x": 9.9, "y": 8.9}, {"x": 11.9, "y": 8.9}]}, {"id": "interior_door|0", "room_type": "interior_door", "area": 0.5, "floor_polygon": [{"x": 8.4, "y": 15.0}, {"x": 8.4, "y": 14.7}, {"x": 6.7, "y": 14.7}, {"x": 6.7, "y": 15.0}]}, {"id": "interior_door|1", "room_type": "interior_door", "area": 0.1, "floor_polygon": [{"x": 9.0, "y": 4.1}, {"x": 9.0, "y": 4.9}, {"x": 9.1, "y": 4.9}, {"x": 9.1, "y": 4.1}]}, {"id": "interior_door|2", "room_type": "interior_door", "area": 0.4, "floor_polygon": [{"x": 7.5, "y": 3.9}, {"x": 8.8, "y": 3.9}, {"x": 8.8, "y": 3.7}, {"x": 7.5, "y": 3.7}]}, {"id": "interior_door|3", "room_type": "interior_door", "area": 0.2, "floor_polygon": [{"x": 10.0, "y": 9.8}, {"x": 10.0, "y": 10.6}, {"x": 10.3, "y": 10.6}, {"x": 10.3, "y": 9.8}]}, {"id": "interior_door|4", "room_type": "interior_door", "area": 0.3, "floor_polygon": [{"x": 8.9, "y": 10.7}, {"x": 8.9, "y": 11.0}, {"x": 9.8, "y": 11.0}, {"x": 9.8, "y": 10.7}]}, {"id": "interior_door|5", "room_type": "interior_door", "area": 0.3, "floor_polygon": [{"x": 7.4, "y": 11.0}, {"x": 8.4, "y": 11.0}, {"x": 8.4, "y": 10.7}, {"x": 7.4, "y": 10.7}]}, {"id": "front_door", "room_type": "front_door", "area": 0.3, "floor_polygon": [{"x": 6.0, "y": 7.2}, {"x": 5.8, "y": 7.2}, {"x": 5.8, "y": 8.2}, {"x": 6.0, "y": 8.2}]}]}}

2. {"input": {"room_count": 7, "total_area": 79.5, "spaces": [{"id": "bedroom|0", "room_type": "bedroom", "height": 4.6, "width": 3.1}, {"id": "living_room", "room_type": "living_room", "area": 32.1}, {"id": "bedroom|1", "room_type": "bedroom", "height": 3.3, "width": 2.5}, {"id": "balcony", "room_type": "balcony", "height": 1.6, "width": 3.7}, {"id": "bedroom|2", "room_type": "bedroom", "height": 2.5, "width": 2.9}, {"id": "kitchen", "room_type": "kitchen", "height": 1.8, "width": 4.7}, {"id": "bathroom", "room_type": "bathroom", "height": 2.5, "width": 1.5}, {"id": "front_door", "room_type": "front_door", "height": 0.1, "width": 1.1}], "input_graph": {"bedroom|0": ["living_room"], "living_room": ["bedroom|0", "kitchen", "bathroom", "bedroom|1", "balcony", "bedroom|2", "front_door"], "bedroom|1": ["living_room"], "balcony": ["living_room"], "bedroom|2": ["living_room"], "kitchen": ["living_room"], "bathroom": ["living_room"], "front_door": ["living_room"]}}}
   {"output": {"room_count": 7, "total_area": 79.5, "spaces": [{"id": "bedroom|0", "room_type": "bedroom", "area": 14.1, "floor_polygon": [{"x": 4.1, "y": 12.7}, {"x": 7.2, "y": 12.7}, {"x": 7.2, "y": 8.2}, {"x": 4.1, "y": 8.2}]}, {"id": "living_room", "room_type": "living_room", "area": 32.1, "floor_polygon": [{"x": 7.5, "y": 8.2}, {"x": 7.5, "y": 9.1}, {"x": 10.2, "y": 9.1}, {"x": 10.2, "y": 12.7}, {"x": 13.9, "y": 12.7}, {"x": 13.9, "y": 3.9}, {"x": 12.4, "y": 3.9}, {"x": 12.4, "y": 5.1}, {"x": 12.5, "y": 5.1}, {"x": 12.5, "y": 5.4}, {"x": 10.2, "y": 5.4}, {"x": 10.2, "y": 8.2}]}, {"id": "bedroom|1", "room_type": "bedroom", "area": 8.1, "floor_polygon": [{"x": 9.9, "y": 9.4}, {"x": 7.5, "y": 9.4}, {"x": 7.5, "y": 12.7}, {"x": 9.9, "y": 12.7}]}, {"id": "balcony", "room_type": "balcony", "area": 6.0, "floor_polygon": [{"x": 13.9, "y": 14.6}, {"x": 13.9, "y": 13.0}, {"x": 10.2, "y": 13.0}, {"x": 10.2, "y": 14.6}]}, {"id": "bedroom|2", "room_type": "bedroom", "area": 7.1, "floor_polygon": [{"x": 8.2, "y": 7.9}, {"x": 8.2, "y": 5.4}, {"x": 5.3, "y": 5.4}, {"x": 5.3, "y": 7.9}]}, {"id": "kitchen", "room_type": "kitchen", "area": 8.3, "floor_polygon": [{"x": 7.5, "y": 5.1}, {"x": 12.2, "y": 5.1}, {"x": 12.2, "y": 3.4}, {"x": 7.5, "y": 3.4}]}, {"id": "bathroom", "room_type": "bathroom", "area": 3.6, "floor_polygon": [{"x": 8.4, "y": 7.9}, {"x": 9.9, "y": 7.9}, {"x": 9.9, "y": 5.4}, {"x": 8.4, "y": 5.4}]}, {"id": "interior_door|0", "room_type": "interior_door", "area": 0.1, "floor_polygon": [{"x": 7.2, "y": 8.4}, {"x": 7.2, "y": 9.0}, {"x": 7.4, "y": 9.0}, {"x": 7.4, "y": 8.4}]}, {"id": "interior_door|1", "room_type": "interior_door", "area": 0.1, "floor_polygon": [{"x": 10.0, "y": 9.4}, {"x": 10.0, "y": 9.2}, {"x": 9.1, "y": 9.2}, {"x": 9.1, "y": 9.4}]}, {"id": "interior_door|2", "room_type": "interior_door", "area": 0.5, "floor_polygon": [{"x": 14.0, "y": 12.8}, {"x": 10.6, "y": 12.8}, {"x": 10.6, "y": 12.9}, {"x": 14.0, "y": 12.9}]}, {"id": "interior_door|3", "room_type": "interior_door", "area": 0.0, "floor_polygon": [{"x": 8.0, "y": 8.1}, {"x": 8.0, "y": 7.9}, {"x": 7.7, "y": 7.9}, {"x": 7.7, "y": 8.1}]}, {"id": "interior_door|4", "room_type": "interior_door", "area": 0.2, "floor_polygon": [{"x": 11.9, "y": 5.2}, {"x": 10.4, "y": 5.2}, {"x": 10.4, "y": 5.3}, {"x": 11.9, "y": 5.3}]}, {"id": "interior_door|5", "room_type": "interior_door", "area": 0.1, "floor_polygon": [{"x": 9.1, "y": 8.1}, {"x": 9.1, "y": 7.9}, {"x": 8.6, "y": 7.9}, {"x": 8.6, "y": 8.1}]}, {"id": "front_door", "room_type": "front_door", "area": 0.2, "floor_polygon": [{"x": 13.8, "y": 3.9}, {"x": 13.8, "y": 3.7}, {"x": 12.7, "y": 3.7}, {"x": 12.7, "y": 3.9}]}]}}

2. {"input": {"room_count": 7, "total_area": 62.4, "spaces": [{"id": "bedroom|0", "room_type": "bedroom", "area": 5.1}, {"id": "living_room", "room_type": "living_room", "area": 26.3}, {"id": "bathroom", "room_type": "bathroom", "height": 2.0, "width": 1.3}, {"id": "bedroom|1", "room_type": "bedroom", "area": 15.1}, {"id": "bedroom|2", "room_type": "bedroom", "area": 6.3}, {"id": "kitchen", "room_type": "kitchen", "height": 1.3, "width": 2.0}, {"id": "balcony", "room_type": "balcony", "height": 1.4, "width": 2.7}, {"id": "front_door", "room_type": "front_door", "height": 0.3, "width": 1.4}], "input_graph": {"bedroom|0": ["living_room"], "living_room": ["bedroom|0", "kitchen", "bathroom", "bedroom|1", "bedroom|2", "balcony", "front_door"], "bathroom": ["living_room"], "bedroom|1": ["living_room"], "bedroom|2": ["living_room"], "kitchen": ["living_room"], "balcony": ["living_room"], "front_door": ["living_room"]}}}
   {"output": {"room_count": 7, "total_area": 62.4, "spaces": [{"id": "bedroom|0", "room_type": "bedroom", "area": 5.1, "floor_polygon": [{"x": 6.2, "y": 5.6}, {"x": 6.2, "y": 7.5}, {"x": 8.8, "y": 7.5}, {"x": 8.8, "y": 5.5}, {"x": 7.1, "y": 5.5}, {"x": 7.1, "y": 5.6}]}, {"id": "living_room", "room_type": "living_room", "area": 26.3, "floor_polygon": [{"x": 11.0, "y": 5.5}, {"x": 11.0, "y": 7.9}, {"x": 8.1, "y": 7.9}, {"x": 8.1, "y": 8.7}, {"x": 11.0, "y": 8.7}, {"x": 11.0, "y": 11.1}, {"x": 11.0, "y": 11.1}, {"x": 11.0, "y": 12.4}, {"x": 14.3, "y": 12.4}, {"x": 14.3, "y": 5.2}, {"x": 12.7, "y": 5.2}, {"x": 12.7, "y": 5.5}]}, {"id": "bathroom", "room_type": "bathroom", "area": 2.6, "floor_polygon": [{"x": 10.5, "y": 7.5}, {"x": 10.5, "y": 5.5}, {"x": 9.2, "y": 5.5}, {"x": 9.2, "y": 7.5}]}, {"id": "bedroom|1", "room_type": "bedroom", "area": 15.1, "floor_polygon": [{"x": 3.7, "y": 11.5}, {"x": 3.7, "y": 12.3}, {"x": 7.6, "y": 12.3}, {"x": 7.6, "y": 11.1}, {"x": 7.7, "y": 11.1}, {"x": 7.7, "y": 7.9}, {"x": 4.4, "y": 7.9}, {"x": 4.4, "y": 11.5}]}, {"id": "bedroom|2", "room_type": "bedroom", "area": 6.3, "floor_polygon": [{"x": 8.1, "y": 11.1}, {"x": 8.2, "y": 11.1}, {"x": 8.2, "y": 11.7}, {"x": 10.5, "y": 11.7}, {"x": 10.5, "y": 11.1}, {"x": 10.5, "y": 11.1}, {"x": 10.5, "y": 9.1}, {"x": 8.1, "y": 9.1}]}, {"id": "kitchen", "room_type": "kitchen", "area": 2.7, "floor_polygon": [{"x": 10.2, "y": 5.1}, {"x": 12.2, "y": 5.1}, {"x": 12.2, "y": 3.7}, {"x": 10.2, "y": 3.7}]}, {"id": "balcony", "room_type": "balcony", "area": 3.9, "floor_polygon": [{"x": 14.3, "y": 12.9}, {"x": 11.6, "y": 12.9}, {"x": 11.6, "y": 14.3}, {"x": 14.3, "y": 14.3}]}, {"id": "interior_door|0", "room_type": "interior_door", "area": 0.1, "floor_polygon": [{"x": 8.6, "y": 7.8}, {"x": 8.6, "y": 7.5}, {"x": 8.3, "y": 7.5}, {"x": 8.3, "y": 7.8}]}, {"id": "interior_door|1", "room_type": "interior_door", "area": 0.2, "floor_polygon": [{"x": 10.0, "y": 7.8}, {"x": 10.0, "y": 7.5}, {"x": 9.4, "y": 7.5}, {"x": 9.4, "y": 7.8}]}, {"id": "interior_door|2", "room_type": "interior_door", "area": 0.1, "floor_polygon": [{"x": 8.0, "y": 8.1}, {"x": 7.7, "y": 8.1}, {"x": 7.7, "y": 8.6}, {"x": 8.0, "y": 8.6}]}, {"id": "interior_door|3", "room_type": "interior_door", "area": 0.2, "floor_polygon": [{"x": 10.5, "y": 8.8}, {"x": 9.7, "y": 8.8}, {"x": 9.7, "y": 9.1}, {"x": 10.5, "y": 9.1}]}, {"id": "interior_door|4", "room_type": "interior_door", "area": 0.3, "floor_polygon": [{"x": 11.2, "y": 5.4}, {"x": 12.4, "y": 5.4}, {"x": 12.4, "y": 5.1}, {"x": 11.2, "y": 5.1}]}, {"id": "interior_door|5", "room_type": "interior_door", "area": 0.7, "floor_polygon": [{"x": 11.7, "y": 12.8}, {"x": 14.2, "y": 12.8}, {"x": 14.2, "y": 12.5}, {"x": 11.7, "y": 12.5}]}, {"id": "front_door", "room_type": "front_door", "area": 0.4, "floor_polygon": [{"x": 14.1, "y": 5.1}, {"x": 14.1, "y": 4.9}, {"x": 12.7, "y": 4.9}, {"x": 12.7, "y": 5.1}]}]}}

"""
    
class FloorplanGenerator:
    def __init__(
        self,
        model_name_or_path="meta-llama/Llama-3.3-70B-Instruct",
        lora_adapter_path=None,
        dataset_name_or_path="datasets/rplan_converted",
        test_split="test",
        test_range=None,
        max_new_tokens=4096,
        batch_size=32,
        device="cuda",
        output_dir="outputs",
        use_sampling=True,
        few_shot_text=None,
        few_shot_path=None
    ):
        self.model_name_or_path = model_name_or_path
        self.enable_lora = lora_adapter_path
        self.lora_adapter_path = lora_adapter_path
        self.dataset_name_or_path = dataset_name_or_path
        self.test_split = test_split
        self.max_new_tokens = max_new_tokens
        self.batch_size = batch_size
        self.device = device
        self.output_dir = output_dir
        self.test_range_start = 0

        # Resolve few-shot text: path > provided text > constant
        resolved_few_shot = None
        try:
            if few_shot_path and os.path.exists(few_shot_path):
                with open(few_shot_path, "r", encoding="utf-8") as f:
                    resolved_few_shot = f.read().strip()
            elif few_shot_text is not None:
                resolved_few_shot = str(few_shot_text).strip()
            else:
                resolved_few_shot = FEW_SHOT_EXAMPLES.strip()
        except Exception:
            resolved_few_shot = FEW_SHOT_EXAMPLES.strip()
        self.few_shot_text = resolved_few_shot

        self.model = LLM(
            model=self.model_name_or_path,
            tensor_parallel_size=4,
            device=self.device,
            enable_lora=self.enable_lora,
            max_lora_rank=256
        )
        if self.enable_lora:
            self.lora_request = LoRARequest(
                "floorplan_adapter", 1, self.lora_adapter_path
            )
        else:
            self.lora_request = None

        if use_sampling:
            self.sampling_params = SamplingParams(
                max_tokens=self.max_new_tokens,
                temperature=0.7,
                top_p=0.9,
                n=10,
                best_of=10
            )
        else:
            self.sampling_params = SamplingParams(
                max_tokens=self.max_new_tokens,
                temperature=0.7,
                top_p=0.9
            )
        
        # Store use_sampling for later use
        self.use_sampling = use_sampling

        self.dataset = load_from_disk(self.dataset_name_or_path)[self.test_split]
        if test_range:
            try:
                self.test_range_start, self.test_range_end = map(int, test_range.split(","))
                self.test_range_start = self.test_range_start - 1
                self.dataset = self.dataset.select(range(self.test_range_start, self.test_range_end))
            except Exception as e:
                print("Invalid test_range format. Expected format: 'start,end' (e.g., '1,101').")
        self.total_examples = len(self.dataset)
        os.makedirs(self.output_dir, exist_ok=True)
    
    def _select_least(self, candidates, input_prompt):
        """
        Select the best candidate by prioritizing:
        1. JSON validity (valid JSON first)
        2. Minimum total_overlap_area 
        3. Minimum compatibility_score
        """
        def _evaluate_candidate(candidate):
            # First priority: JSON validity
            try:
                output_json = extract_output_json(candidate.text)
                if not output_json:  # Invalid or empty JSON
                    return (1, float('inf'), float('inf'))
                json_invalid = 0
            except Exception:
                return (1, float('inf'), float('inf'))  
            
            # Second priority: total overlap area
            try:
                analysis = FeedbackGenerator.analyze(output_json, input_prompt)
                overlap_area = analysis.get('total_overlap_area', float('inf'))
            except Exception:
                overlap_area = float('inf')
            
            # Third priority: compatibility score
            try:
                input_graph = RPLANGraph.from_ds2d(output_json)
                expected_graph = RPLANGraph.from_labeled_adjacency(
                    input_prompt.get("input_graph", {})
                )
                compatibility_score = input_graph.compatibility_score(expected_graph)
            except Exception:
                compatibility_score = float('inf')
            
            return (json_invalid, overlap_area, compatibility_score)
        
        return min(candidates, key=_evaluate_candidate)

    def _build_prompt(self, sample):
        user_payload = sample.get("prompt", "{}")
        if isinstance(user_payload, bytes):
            user_payload = user_payload.decode("utf-8", errors="ignore")
        if self.few_shot_text:
            user_payload = f"{self.few_shot_text}\n{user_payload}"
        prompt = (
            f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n"
            f"{SYSTEM_PROMPT}<|eot_id|><|start_header_id|>user<|end_header_id|>\n"
            f"{user_payload}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n"
        )
        return prompt

    def generate_floorplans(self):
        for i in tqdm(range(0, self.total_examples, self.batch_size), desc="Generating floorplans"):
            raw_batch = self.dataset[i: i + self.batch_size]
            samples = [dict(zip(raw_batch.keys(), t)) for t in zip(*raw_batch.values())]
            batch_prompts = [self._build_prompt(sample) for sample in samples]

            outputs = self.model.generate(
                batch_prompts,
                self.sampling_params,
                lora_request=self.lora_request
            )

            for idx, sample in enumerate(samples):
                input_prompt = ast.literal_eval(sample.get("prompt", "{}"))
                input_prompt = input_prompt["input"]
                if self.use_sampling:
                    output_json = self._select_least(outputs[idx].outputs, input_prompt)
                    output_json = extract_output_json(output_json.text)
                else:
                    generated_text = outputs[idx].outputs[0]
                    output_json = extract_output_json(generated_text.text)

                sample_dir = os.path.join(self.output_dir, str(i + idx + self.test_range_start))
                os.makedirs(sample_dir, exist_ok=True)

                ground_truth_dir = os.path.join(sample_dir, "analysis")
                os.makedirs(ground_truth_dir, exist_ok=True)
                
                with open(os.path.join(sample_dir, "prompt.json"), "w", encoding="utf-8") as f:
                    json.dump(input_prompt, f, indent=4)
                with open(os.path.join(ground_truth_dir, "sample.json"), "w", encoding="utf-8") as f:
                    json.dump(sample, f, indent=4)
                with open(os.path.join(sample_dir, f"0.json"), "w", encoding="utf-8") as f:
                    json.dump(output_json, f, indent=4)
