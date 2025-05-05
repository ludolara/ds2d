from datasets import load_from_disk
from trl import GRPOConfig, GRPOTrainer
from dotenv import load_dotenv
import wandb

load_dotenv()
wandb.init(project="floorplans", mode="offline")

dataset = load_from_disk("datasets/tldr")["train"]

def reward_len(completions, **kwargs):
    return [-abs(20 - len(completion)) for completion in completions]

training_args = GRPOConfig(
    output_dir="output/test-GRPO",
    bf16=True,
    logging_steps=10,
    save_steps=100,
    report_to="wandb"
)

trainer = GRPOTrainer(
    model="Qwen/Qwen2-0.5B-Instruct",
    reward_funcs=reward_len,
    args=training_args,
    train_dataset=dataset,
)
trainer.train()
