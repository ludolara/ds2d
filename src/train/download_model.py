import os
from huggingface_hub import snapshot_download
from datasets import load_dataset
from dotenv import load_dotenv
load_dotenv()

# model_repo_id = "meta-llama/Llama-3.1-8B-Instruct"  
# local_dir = "models/Llama-3.1-8B-Instruct"

# snapshot_download(repo_id=model_repo_id, local_dir=local_dir)
# print(f"Model downloaded to: {local_dir}")

procthor = load_dataset("ludolara/rplan")
print(f"Dataset downloaded to cache directory")
procthor.save_to_disk("hf_datasets/rplan")
