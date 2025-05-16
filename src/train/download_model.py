import os
from huggingface_hub import snapshot_download
from datasets import load_dataset
from dotenv import load_dotenv
load_dotenv()

model_repo_id = "meta-llama/Llama-4-Scout-17B-16E-Instruct"  

snapshot_download(repo_id=model_repo_id, local_dir="models/Llama-4-Scout-17B-16E-Instruct")
print(f"Model downloaded to: models/Llama-4-Scout-17B-16E-Instruct")

# procthor = load_dataset("oops-all-pals/rplan_converted")
# print(f"Dataset downloaded to cache directory: datasets/")
# procthor.save_to_disk("hf_datasets/rplan_converted")



