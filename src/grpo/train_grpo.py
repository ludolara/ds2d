import argparse
from datasets import load_from_disk
from src.grpo.reward_calculator import RewardCalculator
from src.utils import build_prompt
from trl import GRPOConfig, GRPOTrainer
from peft import LoraConfig, TaskType
from dotenv import load_dotenv
import wandb
import os
import glob

load_dotenv()
wandb.init(project="floorplans", mode="offline")

def main():    
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=str, default="output/llama4-GRPO", help="Folder to save the model")
    parser.add_argument("--model", type=str, default="models/Llama-4-Scout-17B-16E-Instruct", help="Model name")
    parser.add_argument("--dataset", type=str, default="hf_datasets/rplan_converted_no_doors", help="Dataset name")
    parser.add_argument("--vllm_server_host", type=str, default="", help="The server IP")
    parser.add_argument("--num_eval_examples", type=int, default=2, help="Number of examples to use for evaluation")
    parser.add_argument("--no_eval", action="store_true", help="Disable evaluation during training")
    # LoRA specific arguments
    parser.add_argument("--use_lora", action="store_true", help="Enable LoRA fine-tuning")
    parser.add_argument("--lora_r", type=int, default=256, help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=512, help="LoRA alpha parameter")
    parser.add_argument("--lora_dropout", type=float, default=0.1, help="LoRA dropout")
    
    args = parser.parse_args()

    # Configure LoRA if enabled
    peft_config = None
    if args.use_lora:
        peft_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        )

    dataset = load_from_disk(args.dataset)
    
    train_dataset = (
        dataset["train"]
        .rename_column("input", "prompt")
        .map(lambda x: {"prompt": build_prompt(x["prompt"])})
    )
    
    eval_dataset = None
    if "validation" in dataset:
        eval_dataset = (
            dataset["validation"]
            .select(range(min(args.num_eval_examples, len(dataset["validation"])))) 
            .rename_column("input", "prompt")
            .map(lambda x: {"prompt": build_prompt(x["prompt"])})
        )
    elif "test" in dataset:
        eval_dataset = (
            dataset["test"]
            .select(range(min(args.num_eval_examples, len(dataset["test"])))) 
            .rename_column("input", "prompt")
            .map(lambda x: {"prompt": build_prompt(x["prompt"])})
        )
    
    do_eval = not args.no_eval

    training_args = GRPOConfig(
        output_dir=args.output,
        # per_device_train_batch_size=2,

        per_device_train_batch_size=8,
        # num_generations=16,
        # gradient_accumulation_steps=4,
        
        max_prompt_length=4096,
        max_completion_length=4096,
        bf16=True,
        gradient_checkpointing=True,
        # logging_steps=50,
        # save_steps=100,
        logging_steps=1,
        save_steps=4,
        save_total_limit=1,
        # save_only_model=True, #remove this line to save the entire trainer
        report_to="wandb",
        use_vllm=True,
        vllm_server_host=args.vllm_server_host,
        resume_from_checkpoint=True, # resume from last checkpoint
        # Evaluation configuration
        do_eval=do_eval,
        eval_strategy="steps" if do_eval else "no",
        eval_steps=4 if do_eval else None, 
        per_device_eval_batch_size=2,
        log_level="info",
    )

    reward_calculator = RewardCalculator()
    reward_funcs = reward_calculator.make_reward_funcs()

    trainer = GRPOTrainer(
        model=args.model, 
        args=training_args, 
        reward_funcs=reward_funcs, 
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        peft_config=peft_config,
    )
    
    checkpoint_dir = None
    if os.path.exists(args.output):
        checkpoint_pattern = os.path.join(args.output, "checkpoint-*")
        checkpoints = glob.glob(checkpoint_pattern)
        if checkpoints:
            latest_checkpoint = max(checkpoints, key=lambda x: int(x.split('-')[-1]))
            checkpoint_dir = latest_checkpoint
            print(f"Resuming from checkpoint: {checkpoint_dir}")
        else:
            print("No checkpoints found, starting fresh training")
    else:
        print("Output directory doesn't exist, starting fresh training")
    
    trainer.train(resume_from_checkpoint=checkpoint_dir)
    trainer.save_model()

if __name__=="__main__":
    main()