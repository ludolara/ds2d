import os
import json
from tqdm import tqdm
from dotenv import load_dotenv
from .feedback_generator import FeedbackGenerator
from src.pred.extract_output_json import extract_output_json
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from datasets import load_from_disk
import torch

load_dotenv()
CACHE_DIR = os.environ.get("TRANSFORMERS_CACHE")
    
class FloorplanGenerator:
    def __init__(
        self,
        model_name_or_path="meta-llama/Llama-3.3-70B-Instruct",
        lora_adapter_path=None,
        test_split="test",
        test_range=None,
        max_new_tokens=2048,
        batch_size=2,
        device="cuda",
        output_dir="outputs"
    ):
        self.model_name_or_path = model_name_or_path
        self.lora_adapter_path = lora_adapter_path
        self.test_split = test_split
        self.max_new_tokens = max_new_tokens
        self.batch_size = batch_size
        self.device = device
        self.output_dir = output_dir
        self.test_range_start = 0

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name_or_path, 
            cache_dir=CACHE_DIR, 
            device_map="auto"
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name_or_path,
            load_in_8bit=True,
            device_map="auto",
            cache_dir=CACHE_DIR
        )
        self.model = PeftModel.from_pretrained(self.model, self.lora_adapter_path, device_map="auto")
        self.model.eval()
        self.model.half()
        # if hasattr(torch, 'compile'):
        #     self.model = torch.compile(self.model)

        # self.dataset = load_dataset("ludolara/DStruct2Design")[self.test_split]
        self.dataset = load_from_disk("datasets/rplan_converted")[self.test_split]
        if test_range:
            try:
                self.test_range_start, self.test_range_end = map(int, test_range.split(","))
                self.test_range_start = self.test_range_start - 1
                self.dataset = self.dataset.select(range(self.test_range_start, self.test_range_end))
            except Exception as e:
                print("Invalid test_range format. Expected format: 'start,end' (e.g., '1,1000'). Using full dataset.")
        self.total_examples = len(self.dataset)
        os.makedirs(self.output_dir, exist_ok=True)

    def _create_input(self, sample, is_str=True):
        inp = {
            "room_count": sample.get("room_count"),
            "total_area": sample.get("total_area"),
            "room_types": sample.get("room_types"),
            "rooms": [
                {
                    "room_type": room.get("room_type"),
                    "width": room.get("width"),
                    "height": room.get("height"),
                    "is_regular": room.get("is_regular"),
                }
                for room in sample.get("rooms", [])
            ],
            # "edges": sample.get("edges", []),
        }
        if is_str:
            return str({"input": inp})
        else:
            return inp

    def _build_prompt(self, sample):
        system_prompt = (
            "you are to generate a floor plan in a JSON structure. "
            "you have to satisfy the adjacency constraints given as pairs of neighboring rooms; "
            "two connecting rooms are presented as (room_type1 room_id1, room_type2 room_id2). "
            "you also need to satisfy additional constraints given by the user."
        )
        prompt = (
            f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n"
            f"{system_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>\n"
            f"{self._create_input(sample)}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n"
        )
        return prompt

    def _build_re_prompt(self, sample, history=None):
        system_prompt = (
            "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\nyou are to generate a floor plan in a JSON structure. "
            "you have to satisfy the adjacency constraints given as pairs of neighboring rooms; "
            "two connecting rooms are presented as (room_type1 room_id1, room_type2 room_id2). "
            "you also need to satisfy additional constraints given by the user.\n"
        )

        history_prompt = ""
        if history:
            for i, attempt in enumerate(history):
                history_prompt += f"- Attempt {i + 1}:\n"
                history_prompt += f"    Output: {json.dumps(attempt['output'])}\n"
                history_prompt += f"    Feedback: {attempt.get('feedback', '[No feedback provided]')}\n"

            history_prompt = (
                f"<|eot_id|><|start_header_id|>user<|end_header_id|>\n"
                f"Below are details of your past attempts:\n{history_prompt}\n"
                f"This will be the attempt number {len(history) + 1} to generate a valid floor plan. Please take all the above into account and try again.\n "
            )
        
        re_prompt = (
            f"{system_prompt}"
            f"{history_prompt}"
            f"<|eot_id|><|start_header_id|>user<|end_header_id|>\n{self._create_input(sample)}"
            f"<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n"
        )

        return re_prompt

    def generate_floorplans(self):
        for i in tqdm(range(0, self.total_examples, self.batch_size), desc="Generating floorplans"):
            raw_batch = self.dataset[i: i + self.batch_size]
            samples = [dict(zip(raw_batch.keys(), t)) for t in zip(*raw_batch.values())]
            prompts = [self._build_prompt(sample) for sample in samples]

            inputs = self.tokenizer(prompts, return_tensors="pt", padding=True, truncation=True)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            with torch.no_grad():
                outputs = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens)

            for idx, output in enumerate(outputs):
                sample_idx = i + idx + self.test_range_start
                sample_dir = os.path.join(self.output_dir, str(sample_idx))
                os.makedirs(sample_dir, exist_ok=True)

                generated_text = self.tokenizer.decode(output, skip_special_tokens=True)
                output_json = extract_output_json(generated_text)

                prompt_path = os.path.join(sample_dir, "prompt.json")
                with open(prompt_path, "w", encoding="utf-8") as f:
                    json.dump(self._create_input(samples[idx], is_str=False), f, indent=4)

                output_path = os.path.join(sample_dir, "0.json")
                with open(output_path, "w", encoding="utf-8") as f:
                    json.dump(output_json, f, indent=4)

    def generate_floorplans_with_feedback(self, feedback_iterations=5):
        for i in tqdm(range(0, self.total_examples, self.batch_size), desc="Generating floorplans with feedback"):
            raw_batch = self.dataset[i: i + self.batch_size]
            samples = [dict(zip(raw_batch.keys(), t)) for t in zip(*raw_batch.values())]
            current_prompts = [self._build_prompt(sample) for sample in samples]
            resolved = [False] * len(samples)
            
            histories = [[] for _ in range(len(samples))]

            for iteration in range(feedback_iterations):
                unresolved_indices = [idx for idx, done in enumerate(resolved) if not done]
                if not unresolved_indices:
                    break  

                batch_prompts = [current_prompts[idx] for idx in unresolved_indices]
                inputs = self.tokenizer(batch_prompts, return_tensors="pt", padding=True, truncation=True)
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                with torch.no_grad():
                    with torch.amp.autocast(self.device):
                        outputs = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens)

                for pos, idx in enumerate(unresolved_indices):
                    generated_text = self.tokenizer.decode(outputs[pos], skip_special_tokens=True)
                    output_json = extract_output_json(generated_text)

                    sample_dir = os.path.join(self.output_dir, str(i + idx + self.test_range_start))
                    sample_dir_feedback = os.path.join(sample_dir, "feedback")
                    os.makedirs(sample_dir, exist_ok=True)
                    os.makedirs(sample_dir_feedback, exist_ok=True)

                    input_prompt = self._create_input(samples[idx], is_str=False)
                    with open(os.path.join(sample_dir, "prompt.json"), "w", encoding="utf-8") as f:
                        json.dump(input_prompt, f, indent=4)
                    with open(os.path.join(sample_dir, f"0.json"), "w", encoding="utf-8") as f:
                        json.dump(output_json, f, indent=4)
                    with open(os.path.join(sample_dir_feedback, f"iteration_{iteration}.json"), "w", encoding="utf-8") as f:
                        json.dump(output_json, f, indent=4)

                    overlap_metrics = FeedbackGenerator.analyze(output_json, input_prompt)
                    
                    current_feedback = ""
                    if overlap_metrics["is_overlapping"] or not overlap_metrics["is_valid_json"] or not overlap_metrics["room_count"]["match"] or not overlap_metrics["room_types"]["match"] or not overlap_metrics["total_area"]["match"]:
                        current_feedback += FeedbackGenerator.create_feedback(overlap_metrics)
                    else:
                        resolved[idx] = True
                        current_feedback = "No issues detected."

                    attempt = {
                        "output": output_json,
                        "iteration": iteration,
                        "feedback": current_feedback,
                        "metrics": overlap_metrics
                    }
                    histories[idx].append(attempt)

                    if not resolved[idx]:
                        new_prompt = self._build_re_prompt(
                            samples[idx],
                            history=histories[idx]
                        )
                        current_prompts[idx] = new_prompt
                        print(new_prompt)
                    
                    with open(os.path.join(sample_dir_feedback, "feedback.json"), "w", encoding="utf-8") as f:
                        filtered_history = [
                            {key: value for key, value in entry.items() if key != "output"}
                            for entry in histories[idx]
                        ]
                        json.dump(filtered_history, f, indent=4)

                if all(resolved):
                    break
