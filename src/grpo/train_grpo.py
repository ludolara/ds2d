import argparse
from datasets import load_from_disk
from src.utils import build_prompt
from src.pred.extract_output_json import extract_output_json
from src.pred.feedback_generator import FeedbackGenerator
from trl import GRPOConfig, GRPOTrainer
from dotenv import load_dotenv
import wandb

load_dotenv()
wandb.init(project="floorplans", mode="offline")

def reward_overlap(completions, **kwargs):
    """
    Reward policy:
    • +1 for no overlap (overlap_percentage == 0)
    • -overlap_ratio for any overlap
    • -1 for invalid JSON or parse errors

    where overlap_ratio = overlap_percentage / 100.0
    """
    rewards = []

    for completion in completions:
        try:
            output_json = extract_output_json(completion)
            stats = FeedbackGenerator.compute_overlap(output_json)
            
            # Penalize invalid JSON
            if not stats.get("is_valid_json", False):
                rewards.append(-1)
                continue

            raw_pct = stats.get("overlap_percentage", 100.0)
            overlap_ratio = raw_pct / 100.0 

            # Give the full reward only if there is absolutely no overlap; otherwise apply a proportional penalty
            if overlap_ratio == 0.0:
                rewards.append(1.5)
            else:
                rewards.append(1-overlap_ratio)

        except Exception as e:
            rewards.append(-1)
    
    print(f"Rewards: {rewards}")
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
        per_device_train_batch_size=1,
        # num_generations=2,
        # gradient_accumulation_steps=4,
        max_prompt_length=4096,
        max_completion_length=4096,
        bf16=True,
        gradient_checkpointing=True,
        logging_steps=25,
        save_steps=50,
        save_total_limit=2,
        save_only_model=True, #remove this line to save the entire trainer
        # load_best_model_at_end=True,
        report_to="wandb",
        use_vllm=True,
        vllm_server_host=args.vllm_server_host,
    )

    trainer = GRPOTrainer(
        model="models/ds2d-Llama-3.3-70B-Instruct", 
        args=training_args, 
        reward_funcs=reward_overlap, 
        train_dataset=dataset
    )
    trainer.train()

if __name__=="__main__":
    main()