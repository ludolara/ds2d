import argparse
from datasets import load_from_disk
from src.pred.extract_output_json import extract_output_json
from src.pred.feedback_generator import FeedbackGenerator
from trl import GRPOConfig, GRPOTrainer
from dotenv import load_dotenv
# import wandb

load_dotenv()
# wandb.init(project="floorplans", mode="offline")
# dataset = load_from_disk("datasets/tldr")["train"]

def main():
    
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--vllm_server_host", type=str, default="", help="The server IP")
    # args = parser.parse_args()

    # dataset = load_from_disk("datasets/tldr")["train"]
    dataset = load_from_disk("hf_datasets/rplan_converted")["train"]
    dataset = dataset.rename_column("input", "prompt")

    def reward_overlap(completions, **kwargs):
        rewards = []
        prompt = kwargs.get("prompt", None)

        for completion in completions:
            try:
                print(completion)
                output_json = extract_output_json(completion)
                stats = FeedbackGenerator.analyze(output_json, prompt)
                overlap = stats["total_overlap_area"]
                rewards.append(-overlap)

            except Exception:
                # parsing or analysis failed → heavy penalty
                rewards.append(-1e6)
        return rewards

    # def reward_len(completions, **kwargs):
    #     print(completions)
    #     return [-abs(20 - len(completion)) for completion in completions]

    training_args = GRPOConfig(
        output_dir="output/test-GRPO_3.3_1",
        per_device_train_batch_size=1,
        num_generations=2,
        gradient_accumulation_steps=4,
        bf16=True,
        gradient_checkpointing=True,
        logging_steps=100,
        save_steps=100,
        report_to="wandb",
        # use_vllm=True,
        # vllm_server_host=args.vllm_server_host,
    )

    trainer = GRPOTrainer(
        model="models/llama-3.1-8B-Instruct",
        args=training_args, 
        # reward_funcs=reward_len, 
        reward_funcs=reward_overlap, 
        train_dataset=dataset
    )
    trainer.train()

if __name__=="__main__":
    main()