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

def main():
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--vllm_server_host", type=str, default="", help="The server IP")
    args = parser.parse_args()
    
    # dataset = load_from_disk("datasets/tldr")["train"]
    # def reward_len(completions, **kwargs):
    #     return [-abs(20 - len(completion)) for completion in completions]

    dataset = load_from_disk("hf_datasets/rplan_converted")["train"]
    dataset = dataset.rename_column("input", "prompt")
    dataset = dataset.map(lambda x: {"prompt": build_prompt(x["prompt"])})

    # def reward_overlap(completions, **kwargs):
    #     rewards = []
    #     prompt = kwargs.get("prompt", None)

    #     for completion in completions:
    #         try:
    #             output_json = extract_output_json(completion)
    #             print(output_json)
    #             stats = FeedbackGenerator.analyze(output_json, prompt)
    #             overlap = stats["total_overlap_area"]
    #             rewards.append(-overlap)

    #         except Exception:
    #             # parsing or analysis failed → heavy penalty
    #             rewards.append(-1e6)
    #     return rewards

    PENALTY_SCORE = -1e6

    def reward_overlap(completions, **kwargs):
        rewards = []
        prompt = kwargs.get("prompt", None)

        for completion in completions:
            try:
                output_json = extract_output_json(completion)
                stats = FeedbackGenerator.analyze(output_json, prompt)

                if not stats.get("is_valid_json", False):
                    rewards.append(PENALTY_SCORE)
                    continue

                overlap = stats.get("total_overlap_area", float("inf"))
                rewards.append(-overlap)

            except Exception:
                rewards.append(PENALTY_SCORE)

        return rewards

    training_args = GRPOConfig(
        output_dir="output/test-GRPO_3.3",
        per_device_train_batch_size=1,
        num_generations=2,
        gradient_accumulation_steps=8,
        max_prompt_length=4096,
        max_completion_length=4096,
        bf16=True,
        gradient_checkpointing=True,
        logging_steps=500,
        save_steps=1000,
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