import argparse
from datasets import load_from_disk
from src.utils import build_prompt
from src.pred.extract_output_json import extract_output_json
from src.pred.feedback_generator import FeedbackGenerator
from src.utils.create_example import build_prompt_llama4
from trl import GRPOConfig, GRPOTrainer
from dotenv import load_dotenv
import wandb

load_dotenv()
wandb.init(project="floorplans", mode="offline")

def _safe_feedback(c, p):
    try:
        # TODO: remove extract_output_json with llama4
        return FeedbackGenerator.grpo_feedback(
            extract_output_json(c),
            p
        )
    except Exception as e:
        print(e)
        return {"is_valid_json": False}

def make_reward_funcs():
    cache = {"stats": None, "key": None}

    def compute_stats(completions, **kwargs):
        key = id(completions[0] + completions[-1])
        if cache["key"] != key:
            cache["stats"] = [
                _safe_feedback(comp, {"room_count": rc, "total_area": ta})
                for comp, rc, ta in zip(
                    completions, 
                    kwargs.get("room_count", []), 
                    kwargs.get("total_area", [])    
                )
            ]
            cache["key"] = key
        return cache["stats"]

    def json_validity_reward(completions, **kwargs):
        stats_list = compute_stats(completions, **kwargs)
        return [
            1.0 if s.get("is_valid_json", False) else 0.0
            for s in stats_list
        ]

    def room_count_reward(completions, **kwargs):
        stats_list = compute_stats(completions, **kwargs)
        rewards = [
            0.0 if not s.get("is_valid_json", False)
            else 1.0 if (r := s.get("room_count", 0.0)) == 1.0 else 1.0 - abs(r - 1.0)
            for s in stats_list
        ]
        # print(rewards)
        return rewards

    def total_area_reward(completions, **kwargs):
        stats_list = compute_stats(completions, **kwargs)
        rewards = [
            0.0 if not s.get("is_valid_json", False)
            else 1.0 if (a := s.get("total_area", 0.0)) == 1.0 else 1.0 - abs(a - 1.0)
            for s in stats_list
        ]
        # print(rewards)
        return rewards

    def overlap_reward(completions, **kwargs):
        stats_list = compute_stats(completions, **kwargs)
        rewards = [
            0.0 if not s.get("is_valid_json", False)
            else 1.0 if (o := s.get("overlap", 1.0)) == 0.0 else 1.0 - o
            for s in stats_list
        ]
        # print(rewards)
        return rewards
    
    def compactness_reward(completions, **kwargs):
        stats_list = compute_stats(completions, **kwargs)
        rewards = [
            0.0 if not s.get("is_valid_json", False)
            else 1.0 if (a := s.get("compactness", 0.0)) == 1.0 else 1.0 - abs(a - 1.0)
            for s in stats_list
        ]
        # print(rewards)
        return rewards

    return [
        json_validity_reward,
        room_count_reward,
        total_area_reward,
        overlap_reward,
        compactness_reward
    ]

def main():    
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=str, default="output/llama4-GRPO", help="Folder to save the model")
    parser.add_argument("--model", type=str, default="models/Llama-4-Scout-17B-16E-Instruct", help="Model name")
    parser.add_argument("--dataset", type=str, default="hf_datasets/rplan_converted_no_doors", help="Dataset name")
    parser.add_argument("--vllm_server_host", type=str, default="", help="The server IP")
    args = parser.parse_args()

    dataset = (
        load_from_disk(args.dataset)["train"]
        .rename_column("input", "prompt")
        .map(lambda x: {"prompt": build_prompt_llama4(x["prompt"])})
    )

    training_args = GRPOConfig(
        output_dir=args.output,
        # per_device_train_batch_size=1,
        # num_generations=16,
        # gradient_accumulation_steps=4,
        max_prompt_length=4096,
        max_completion_length=4096,
        bf16=True,
        gradient_checkpointing=True,
        logging_steps=25,
        save_steps=50,
        save_total_limit=3,
        save_only_model=True, #remove this line to save the entire trainer
        # load_best_model_at_end=True,
        report_to="wandb",
        use_vllm=True,
        vllm_server_host=args.vllm_server_host,
    )

    trainer = GRPOTrainer(
        model=args.model, 
        args=training_args, 
        reward_funcs=make_reward_funcs(), 
        train_dataset=dataset
    )
    trainer.train()

if __name__=="__main__":
    main()