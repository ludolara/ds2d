import os
from dotenv import load_dotenv
from huggingface_hub import snapshot_download
from datasets import load_dataset, load_from_disk

model_repo_id = "meta-llama/Llama-4-Scout-17B-16E-Instruct"  

snapshot_download(repo_id=model_repo_id, local_dir="models/Llama-4-Scout-17B-16E-Instruct")
print(f"Model downloaded to: models/Llama-4-Scout-17B-16E-Instruct")

# procthor = load_dataset("ludolara/DStruct2Design-nightly", cache_dir=cache_dir)
# print(f"Dataset 'ludolara/DStruct2Design-nightly' downloaded to cache directory: datasets/")
# procthor.save_to_disk("datasets/DStruct2Design-nightly")

# procthor = load_from_disk('datasets/rplan_converted')
# print(procthor['train'][60101])


