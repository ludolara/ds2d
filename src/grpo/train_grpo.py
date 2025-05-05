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
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    save_steps=100,
    report_to="wandb"
)

trainer = GRPOTrainer(
    model="models/Llama-3.1-8B-Instruct",
    reward_funcs=reward_len,
    args=training_args,
    train_dataset=dataset,
)
trainer.train()

# import argparse
# from datasets import load_from_disk
# from trl import GRPOTrainer, GRPOConfig

# def main():
    
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--vllm_server_host", type=str, default="", help="The server IP")
#     args = parser.parse_args()

#     # Example dataset from TLDR
#     dataset = load_from_disk("datasets/tldr")["train"]

#     # Dummy reward function: count the number of unique characters in the completions
#     def reward_num_unique_chars(completions, **kwargs):
#         return [len(set(c)) for c in completions]

#     training_args = GRPOConfig(
#         output_dir="test-GRPO",
#         per_device_train_batch_size=1,
#         bf16=True,
#         gradient_checkpointing=True,
#         logging_steps=10,
#         use_vllm=True,
#         vllm_server_host=args.vllm_server_host.replace("ip-", "").replace("-", "."),  # from ip-X-X-X-X to X.X.X.X
#     )

#     trainer = GRPOTrainer(model="models/Llama-3.1-8B-Instruct", args=training_args, reward_funcs=reward_num_unique_chars, train_dataset=dataset)
#     trainer.train()

# if __name__=="__main__":
#     main()