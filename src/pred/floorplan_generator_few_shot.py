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

1. {"input": {"room_count": 6, "total_area": 81.4, "spaces": [{"id": "kitchen", "room_type": "kitchen", "height": 2.5, "width": 2.2}, {"id": "bathroom", "room_type": "bathroom", "height": 1.5, "width": 2.2}, {"id": "living_room", "room_type": "living_room", "area": 39.9}, {"id": "balcony", "room_type": "balcony", "height": 1.1, "width": 7.3}, {"id": "bedroom|0", "room_type": "bedroom", "height": 3.4, "width": 3.5}, {"id": "bedroom|1", "room_type": "bedroom", "height": 3.4, "width": 3.5}, {"id": "front_door", "room_type": "front_door", "height": 0.1, "width": 0.8}], "input_graph": {"kitchen": ["living_room"], "bathroom": ["living_room"], "living_room": ["kitchen", "bathroom", "bedroom|1", "bedroom|0", "front_door"], "balcony": ["bedroom|1"], "bedroom|0": ["living_room"], "bedroom|1": ["living_room", "balcony"], "front_door": ["living_room"]}}}
   {"output": {"room_count": 6, "total_area": 81.4, "spaces": [{"id": "kitchen", "room_type": "kitchen", "area": 5.5, "floor_polygon": [{"x": 6.8, "y": 4.7}, {"x": 9.0, "y": 4.7}, {"x": 9.0, "y": 2.2}, {"x": 6.8, "y": 2.2}]}, {"id": "bathroom", "room_type": "bathroom", "area": 3.5, "floor_polygon": [{"x": 6.8, "y": 6.7}, {"x": 9.0, "y": 6.7}, {"x": 9.0, "y": 5.1}, {"x": 6.8, "y": 5.1}]}, {"id": "living_room", "room_type": "living_room", "area": 39.9, "floor_polygon": [{"x": 9.4, "y": 2.2}, {"x": 9.4, "y": 7.0}, {"x": 5.3, "y": 7.0}, {"x": 5.3, "y": 10.3}, {"x": 12.7, "y": 10.3}, {"x": 12.7, "y": 2.2}]}, {"id": "balcony", "room_type": "balcony", "area": 8.2, "floor_polygon": [{"x": 12.7, "y": 15.8}, {"x": 12.7, "y": 14.6}, {"x": 5.3, "y": 14.6}, {"x": 5.3, "y": 15.8}]}, {"id": "bedroom|0", "room_type": "bedroom", "area": 12.1, "floor_polygon": [{"x": 8.9, "y": 10.8}, {"x": 5.3, "y": 10.8}, {"x": 5.3, "y": 14.2}, {"x": 8.9, "y": 14.2}]}, {"id": "bedroom|1", "room_type": "bedroom", "area": 12.1, "floor_polygon": [{"x": 12.7, "y": 10.8}, {"x": 9.1, "y": 10.8}, {"x": 9.1, "y": 14.2}, {"x": 12.7, "y": 14.2}]}, {"id": "interior_door|0", "room_type": "interior_door", "area": 0.2, "floor_polygon": [{"x": 9.1, "y": 6.0}, {"x": 9.4, "y": 6.0}, {"x": 9.4, "y": 5.3}, {"x": 9.1, "y": 5.3}]}, {"id": "interior_door|1", "room_type": "interior_door", "area": 0.2, "floor_polygon": [{"x": 9.1, "y": 3.7}, {"x": 9.1, "y": 4.5}, {"x": 9.4, "y": 4.5}, {"x": 9.4, "y": 3.7}]}, {"id": "interior_door|2", "room_type": "interior_door", "area": 0.2, "floor_polygon": [{"x": 7.9, "y": 10.4}, {"x": 7.9, "y": 10.7}, {"x": 8.6, "y": 10.7}, {"x": 8.6, "y": 10.4}]}, {"id": "interior_door|3", "room_type": "interior_door", "area": 0.2, "floor_polygon": [{"x": 10.4, "y": 10.7}, {"x": 10.4, "y": 10.4}, {"x": 9.6, "y": 10.4}, {"x": 9.6, "y": 10.7}]}, {"id": "interior_door|4", "room_type": "interior_door", "area": 0.2, "floor_polygon": [{"x": 10.5, "y": 14.3}, {"x": 9.6, "y": 14.3}, {"x": 9.6, "y": 14.6}, {"x": 10.5, "y": 14.6}]}, {"id": "front_door", "room_type": "front_door", "area": 0.1, "floor_polygon": [{"x": 5.3, "y": 6.9}, {"x": 6.2, "y": 6.9}, {"x": 6.2, "y": 6.8}, {"x": 5.3, "y": 6.8}]}]}}

2. {"input": {"room_count": 6, "total_area": 58.3, "spaces": [{"id": "bedroom|0", "room_type": "bedroom", "height": 3.4, "width": 3.2}, {"id": "kitchen", "room_type": "kitchen", "height": 2.6, "width": 1.6}, {"id": "bathroom", "room_type": "bathroom", "height": 1.6, "width": 2.0}, {"id": "bedroom|1", "room_type": "bedroom", "height": 3.1, "width": 3.7}, {"id": "living_room", "room_type": "living_room", "area": 25.4}, {"id": "balcony", "room_type": "balcony", "height": 0.8, "width": 3.7}, {"id": "front_door", "room_type": "front_door", "height": 0.7, "width": 0.1}], "input_graph": {"bedroom|0": ["living_room"], "kitchen": ["living_room"], "bathroom": ["living_room"], "bedroom|1": ["balcony", "living_room"], "living_room": ["bedroom|0", "kitchen", "bathroom", "bedroom|1", "front_door"], "balcony": ["bedroom|1"], "front_door": ["living_room"]}}}
   {"output": {"room_count": 6, "total_area": 58.3, "spaces": [{"id": "bedroom|0", "room_type": "bedroom", "area": 10.9, "floor_polygon": [{"x": 3.4, "y": 4.1}, {"x": 3.4, "y": 7.5}, {"x": 6.6, "y": 7.5}, {"x": 6.6, "y": 4.1}]}, {"id": "kitchen", "room_type": "kitchen", "area": 4.2, "floor_polygon": [{"x": 13.0, "y": 6.8}, {"x": 14.6, "y": 6.8}, {"x": 14.6, "y": 4.1}, {"x": 13.0, "y": 4.1}]}, {"id": "bathroom", "room_type": "bathroom", "area": 3.3, "floor_polygon": [{"x": 5.4, "y": 7.8}, {"x": 3.4, "y": 7.8}, {"x": 3.4, "y": 9.4}, {"x": 5.4, "y": 9.4}]}, {"id": "bedroom|1", "room_type": "bedroom", "area": 11.5, "floor_polygon": [{"x": 7.1, "y": 9.7}, {"x": 3.4, "y": 9.7}, {"x": 3.4, "y": 12.8}, {"x": 7.1, "y": 12.8}]}, {"id": "living_room", "room_type": "living_room", "area": 25.4, "floor_polygon": [{"x": 5.7, "y": 7.8}, {"x": 5.7, "y": 9.4}, {"x": 8.4, "y": 9.4}, {"x": 8.4, "y": 7.5}, {"x": 14.6, "y": 7.5}, {"x": 14.6, "y": 7.0}, {"x": 12.7, "y": 7.0}, {"x": 12.7, "y": 4.1}, {"x": 6.9, "y": 4.1}, {"x": 6.9, "y": 7.8}]}, {"id": "balcony", "room_type": "balcony", "area": 2.9, "floor_polygon": [{"x": 7.1, "y": 13.1}, {"x": 3.4, "y": 13.1}, {"x": 3.4, "y": 13.9}, {"x": 7.1, "y": 13.9}]}, {"id": "interior_door|0", "room_type": "interior_door", "area": 0.2, "floor_polygon": [{"x": 12.8, "y": 6.5}, {"x": 12.9, "y": 6.5}, {"x": 12.9, "y": 4.9}, {"x": 12.8, "y": 4.9}]}, {"id": "interior_door|1", "room_type": "interior_door", "area": 0.1, "floor_polygon": [{"x": 6.0, "y": 7.6}, {"x": 6.0, "y": 7.7}, {"x": 6.8, "y": 7.7}, {"x": 6.8, "y": 7.6}]}, {"id": "interior_door|2", "room_type": "interior_door", "area": 0.1, "floor_polygon": [{"x": 5.5, "y": 9.3}, {"x": 5.6, "y": 9.3}, {"x": 5.6, "y": 8.8}, {"x": 5.5, "y": 8.8}]}, {"id": "interior_door|3", "room_type": "interior_door", "area": 0.1, "floor_polygon": [{"x": 7.2, "y": 9.5}, {"x": 6.5, "y": 9.5}, {"x": 6.5, "y": 9.6}, {"x": 7.2, "y": 9.6}]}, {"id": "interior_door|4", "room_type": "interior_door", "area": 0.2, "floor_polygon": [{"x": 5.8, "y": 13.0}, {"x": 5.8, "y": 12.9}, {"x": 4.2, "y": 12.9}, {"x": 4.2, "y": 13.0}]}, {"id": "front_door", "room_type": "front_door", "area": 0.1, "floor_polygon": [{"x": 14.7, "y": 7.0}, {"x": 14.7, "y": 7.7}, {"x": 14.8, "y": 7.7}, {"x": 14.8, "y": 7.0}]}]}}

3. {"input": {"room_count": 6, "total_area": 62.0, "spaces": [{"id": "kitchen", "room_type": "kitchen", "height": 2.6, "width": 3.2}, {"id": "bathroom", "room_type": "bathroom", "height": 2.6, "width": 2.9}, {"id": "living_room", "room_type": "living_room", "height": 2.5, "width": 7.6}, {"id": "bedroom|0", "room_type": "bedroom", "height": 3.2, "width": 3.7}, {"id": "bedroom|1", "room_type": "bedroom", "height": 3.2, "width": 3.6}, {"id": "balcony", "room_type": "balcony", "height": 1.0, "width": 3.7}, {"id": "front_door", "room_type": "front_door", "height": 0.1, "width": 0.8}], "input_graph": {"kitchen": ["living_room"], "bathroom": ["living_room"], "living_room": ["kitchen", "bathroom", "bedroom|1", "bedroom|0", "front_door"], "bedroom|0": ["balcony", "living_room"], "bedroom|1": ["living_room"], "balcony": ["bedroom|0"], "front_door": ["living_room"]}}}
   {"output": {"room_count": 6, "total_area": 62.0, "spaces": [{"id": "kitchen", "room_type": "kitchen", "area": 8.4, "floor_polygon": [{"x": 5.2, "y": 3.9}, {"x": 5.2, "y": 6.5}, {"x": 8.4, "y": 6.5}, {"x": 8.4, "y": 3.9}]}, {"id": "bathroom", "room_type": "bathroom", "area": 7.5, "floor_polygon": [{"x": 11.6, "y": 3.9}, {"x": 8.7, "y": 3.9}, {"x": 8.7, "y": 6.5}, {"x": 11.6, "y": 6.5}]}, {"id": "living_room", "room_type": "living_room", "area": 19.2, "floor_polygon": [{"x": 12.8, "y": 9.4}, {"x": 12.8, "y": 6.8}, {"x": 5.2, "y": 6.8}, {"x": 5.2, "y": 9.4}]}, {"id": "bedroom|0", "room_type": "bedroom", "area": 11.8, "floor_polygon": [{"x": 5.2, "y": 12.8}, {"x": 8.9, "y": 12.8}, {"x": 8.9, "y": 9.6}, {"x": 5.2, "y": 9.6}]}, {"id": "bedroom|1", "room_type": "bedroom", "area": 11.3, "floor_polygon": [{"x": 9.2, "y": 9.6}, {"x": 9.2, "y": 12.8}, {"x": 12.8, "y": 12.8}, {"x": 12.8, "y": 9.6}]}, {"id": "balcony", "room_type": "balcony", "area": 3.7, "floor_polygon": [{"x": 5.2, "y": 13.1}, {"x": 5.2, "y": 14.1}, {"x": 8.9, "y": 14.1}, {"x": 8.9, "y": 13.1}]}, {"id": "interior_door|0", "room_type": "interior_door", "area": 0.1, "floor_polygon": [{"x": 9.4, "y": 6.6}, {"x": 8.9, "y": 6.6}, {"x": 8.9, "y": 6.8}, {"x": 9.4, "y": 6.8}]}, {"id": "interior_door|1", "room_type": "interior_door", "area": 0.1, "floor_polygon": [{"x": 6.0, "y": 6.8}, {"x": 6.0, "y": 6.6}, {"x": 5.1, "y": 6.6}, {"x": 5.1, "y": 6.8}]}, {"id": "interior_door|2", "room_type": "interior_door", "area": 0.1, "floor_polygon": [{"x": 6.0, "y": 9.6}, {"x": 6.0, "y": 9.4}, {"x": 5.1, "y": 9.4}, {"x": 5.1, "y": 9.6}]}, {"id": "interior_door|3", "room_type": "interior_door", "area": 0.1, "floor_polygon": [{"x": 9.4, "y": 9.4}, {"x": 9.4, "y": 9.6}, {"x": 10.0, "y": 9.6}, {"x": 10.0, "y": 9.4}]}, {"id": "interior_door|4", "room_type": "interior_door", "area": 0.2, "floor_polygon": [{"x": 7.7, "y": 12.9}, {"x": 6.2, "y": 12.9}, {"x": 6.2, "y": 13.0}, {"x": 7.7, "y": 13.0}]}, {"id": "front_door", "room_type": "front_door", "area": 0.1, "floor_polygon": [{"x": 12.0, "y": 6.8}, {"x": 12.9, "y": 6.8}, {"x": 12.9, "y": 6.6}, {"x": 12.0, "y": 6.6}]}]}}

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
