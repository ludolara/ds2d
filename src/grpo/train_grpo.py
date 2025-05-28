import argparse
from datasets import load_from_disk
from src.grpo.reward_calculator import RewardCalculator
from src.utils import build_prompt
from trl import GRPOConfig, GRPOTrainer
from dotenv import load_dotenv
import wandb

load_dotenv()
wandb.init(project="floorplans", mode="offline")

def main():    
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=str, default="output/llama4-GRPO", help="Folder to save the model")
    parser.add_argument("--model", type=str, default="models/Llama-4-Scout-17B-16E-Instruct", help="Model name")
    parser.add_argument("--dataset", type=str, default="hf_datasets/rplan_converted_no_doors", help="Dataset name")
    parser.add_argument("--vllm_server_host", type=str, default="", help="The server IP")
    args = parser.parse_args()

    train_dataset = (
        load_from_disk(args.dataset)["train"]
        .rename_column("input", "prompt")
        .map(lambda x: {"prompt": build_prompt(x["prompt"])})
    )

    training_args = GRPOConfig(
        output_dir=args.output,
        per_device_train_batch_size=2,
        # num_generations=16,
        # gradient_accumulation_steps=4,
        max_prompt_length=4096,
        max_completion_length=4096,
        bf16=True,
        gradient_checkpointing=True,
        logging_steps=50,
        save_steps=100,
        save_total_limit=3,
        save_only_model=True, #remove this line to save the entire trainer
        report_to="wandb",
        use_vllm=True,
        vllm_server_host=args.vllm_server_host,
    )

    reward_calculator = RewardCalculator()
    reward_funcs = reward_calculator.make_reward_funcs()

    trainer = GRPOTrainer(
        model=args.model, 
        args=training_args, 
        reward_funcs=reward_funcs, 
        train_dataset=train_dataset,
    )
    trainer.train()
    trainer.save_model()

if __name__=="__main__":
    main()