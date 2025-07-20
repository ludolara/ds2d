import argparse
from datasets import load_from_disk
from src.grpo.reward_calculator import RewardCalculator
from src.utils import build_prompt
# from trl import GRPOConfig, GRPOTrainer
from trl import GRPOConfig
from dotenv import load_dotenv
import wandb
# import os
# import glob
from src.grpo.custom_grpo_trainer import CustomGRPOTrainer, BestRewardCallback

load_dotenv()
wandb.init(project="floorplans", mode="offline")

def main():    
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=str, default="output/llama4-GRPO", help="Folder to save the model")
    parser.add_argument("--model", type=str, default="models/Llama-4-Scout-17B-16E-Instruct", help="Model name")
    parser.add_argument("--dataset", type=str, default="hf_datasets/rplan_converted_no_doors", help="Dataset name")
    parser.add_argument("--vllm_server_host", type=str, default="", help="The server IP")
    parser.add_argument("--eval_sample_size", type=int, default=200, help="Number of examples to use for evaluation")
    # parser.add_argument("--eval_sample_size", type=int, default=20, help="Number of examples to use for evaluation")
    parser.add_argument("--no_eval", action="store_true", help="Disable evaluation during training")
    parser.add_argument("--early_stopping_patience", type=int, default=3, help="Early stopping patience")
    
    args = parser.parse_args()

    dataset = load_from_disk(args.dataset)
    
    train_dataset = (
        dataset["train"]
        .map(lambda x: {"prompt": build_prompt(x)})
    )
    
    eval_dataset = None
    if "validation" in dataset:
        eval_dataset = (
            dataset["validation"]
            .map(lambda x: {"prompt": build_prompt(x)})
        )
    
    do_eval = not args.no_eval
    training_args = GRPOConfig(
        output_dir=args.output,
        per_device_train_batch_size=1,
        num_generations=4,
        max_prompt_length=4096,
        max_completion_length=4096,
        bf16=True,
        gradient_checkpointing=True,

        # logging_steps=1,
        ## save_steps=4,
        # eval_steps=4 if do_eval else None, 
        
        logging_steps=50,
        # save_steps=100,
        eval_steps=100 if do_eval else None,
        
        report_to="wandb",
        use_vllm=True,
        vllm_server_host=args.vllm_server_host,
        # resume_from_checkpoint=True,
        do_eval=do_eval,
        eval_strategy="steps" if do_eval else "no",
        per_device_eval_batch_size=1,
        log_level="info",
        warmup_steps=100,
        
        save_strategy="no",  # BestRewardCallback controls saving
        save_total_limit=2,
        save_only_model=True
    )

    reward_calculator = RewardCalculator()
    reward_funcs = reward_calculator.make_reward_funcs()

    # trainer = GRPOTrainer(
    trainer = CustomGRPOTrainer(
        model=args.model, 
        args=training_args, 
        reward_funcs=reward_funcs, 
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        eval_sample_size=args.eval_sample_size,
        callbacks=[
            BestRewardCallback(early_stopping_patience=args.early_stopping_patience),
        ],
    )
    
    # checkpoint_dir = None
    # if os.path.exists(args.output):
    #     checkpoint_pattern = os.path.join(args.output, "checkpoint-*")
    #     checkpoints = glob.glob(checkpoint_pattern)
    #     if checkpoints:
    #         latest_checkpoint = max(checkpoints, key=lambda x: int(x.split('-')[-1]))
    #         checkpoint_dir = latest_checkpoint
    #         print(f"Resuming from checkpoint: {checkpoint_dir}")
    #     else:
    #         print("No checkpoints found, starting fresh training")
    # else:
    #     print("Output directory doesn't exist, starting fresh training")
    # trainer.train(resume_from_checkpoint=checkpoint_dir)

    trainer.train()
    trainer.save_model()

if __name__=="__main__":
    main()
