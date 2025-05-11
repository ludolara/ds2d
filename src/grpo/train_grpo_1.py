import argparse
import json
from datasets import load_from_disk
from src.utils import build_prompt
from src.pred.extract_output_json import extract_output_json
from src.pred.feedback_generator import FeedbackGenerator
from trl import GRPOConfig, GRPOTrainer
from dotenv import load_dotenv
import wandb

load_dotenv()
wandb.init(project="floorplans", mode="offline")

def get_room_count_match(stats) -> int:
    return 1 if stats.get("room_count", {}).get("match", False) else 0

def calculate_overlap_reward(stats) -> float:
    overlap_percentage = stats.get("overlap_percentage", 100.0)
    overlap_ratio = overlap_percentage / 100.0
    return 1 if overlap_ratio == 0.0 else 1 - overlap_ratio

def reward_overlap(completions, prompts, **kwargs):
    rewards = []

    for completion, prompt in zip(completions, prompts):
        try:
            output_json = extract_output_json(completion)
            stats = FeedbackGenerator.analyze(output_json, prompt)
            
            if not stats.get("is_valid_json", False):
                rewards.append([0, 0, 0])
                continue

            room_count_reward = get_room_count_match(stats)
            overlap_reward = calculate_overlap_reward(stats)

            rewards.append([1, room_count_reward, overlap_reward])

        except Exception:
            rewards.append([0, 0, 0])

    return rewards

def main():    
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=str, default="output/ds2d-GRPO_70B", help="Folder to save the model")
    parser.add_argument("--vllm_server_host", type=str, default="", help="The server IP")
    args = parser.parse_args()

    dataset = load_from_disk("hf_datasets/rplan_converted")["train"]
    dataset = dataset.rename_column("input", "prompt")
    dataset = dataset.map(lambda x: {"prompt": build_prompt(x["prompt"])})

    training_args = GRPOConfig(
        output_dir=args.output,
        # per_device_train_batch_size=1,
        num_generations=4,
        # gradient_accumulation_steps=2,
        max_prompt_length=4096,
        max_completion_length=4096,
        bf16=True,
        gradient_checkpointing=True,
        logging_steps=25,
        save_steps=100,
        save_total_limit=2,
        save_only_model=True, #remove this line to save the entire trainer
        # report_to="wandb",
    )

    trainer = GRPOTrainer(
        model="models/ds2d-Llama-3.1-8B-Instruct", 
        args=training_args, 
        reward_funcs=reward_overlap, 
        train_dataset=dataset
    )
    trainer.train()

if __name__=="__main__":
    main()