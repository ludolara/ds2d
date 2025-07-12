import os
import json
from tqdm import tqdm
from dotenv import load_dotenv
from src.dataset_convert.rplan_graph import RPLANGraph
from src.utils import build_prompt
from src.pred.feedback_generator import FeedbackGenerator
from src.pred.extract_output_json import extract_output_json
from datasets import load_from_disk
# from src.utils.constants import SYSTEM_PROMPT, SYSTEM_RE_PROMPT
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

load_dotenv()
CACHE_DIR = os.environ.get("TRANSFORMERS_CACHE")
    
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
        use_sampling=True
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

    # def _build_prompt(self, sample):
    #     prompt = (
    #         f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n"
    #         f"{SYSTEM_PROMPT}<|eot_id|><|start_header_id|>user<|end_header_id|>\n"
    #         f"{sample.get("prompt", {})}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n"
    #     )
    #     return prompt
    
    def _select_least(self, candidates, input_prompt):
        """
        Return the candidate minimizing first total_overlap_area,
        then compatibility_score.
        """
        def _key(cand):
            output = extract_output_json(cand.text)
            analysis = FeedbackGenerator.analyze(output, input_prompt)
            overlap = analysis.get('total_overlap_area', float('inf'))

            try:
                input_graph = RPLANGraph.from_ds2d(output)
                expected_graph = RPLANGraph.from_labeled_adjacency(
                    input_prompt.get("input_graph", {})
                )
                compatibility_score = input_graph.compatibility_score(expected_graph)
            except Exception:
                compatibility_score = float('inf')

            return (overlap, compatibility_score)

        return min(candidates, key=_key)

    def generate_floorplans(self):
        for i in tqdm(range(0, self.total_examples, self.batch_size), desc="Generating floorplans"):
            raw_batch = self.dataset[i: i + self.batch_size]
            samples = [dict(zip(raw_batch.keys(), t)) for t in zip(*raw_batch.values())]
            batch_prompts = [build_prompt(sample) for sample in samples]

            outputs = self.model.generate(
                batch_prompts,
                self.sampling_params,
                lora_request=self.lora_request
            )

            for idx, sample in enumerate(samples):
                input_prompt = json.loads(sample.get("prompt", {}))
                if self.use_sampling:
                    # output_json = self._select_least_overlap(outputs[idx].outputs, create_input(sample, is_str=False))
                    output_json = self._select_least(outputs[idx].outputs, input_prompt)
                    output_json = extract_output_json(output_json.text)
                else:
                    generated_text = outputs[idx].outputs[0]
                    output_json = extract_output_json(generated_text.text)

                sample_dir = os.path.join(self.output_dir, str(i + idx + self.test_range_start))
                os.makedirs(sample_dir, exist_ok=True)

                with open(os.path.join(sample_dir, "prompt.json"), "w", encoding="utf-8") as f:
                    json.dump(input_prompt, f, indent=4)
                with open(os.path.join(sample_dir, f"0.json"), "w", encoding="utf-8") as f:
                    json.dump(output_json, f, indent=4)

