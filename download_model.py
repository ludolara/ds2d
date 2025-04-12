import os
from dotenv import load_dotenv
from huggingface_hub import snapshot_download
from datasets import load_dataset, load_from_disk

model_repo_id = "meta-llama/Llama-3.3-70B-Instruct"  

snapshot_download(repo_id=model_repo_id, local_dir="models/Llama-3.3-70B-Instruct")
print(f"Model downloaded to: models/Llama-3.3-70B-Instruct")

# procthor = load_dataset("ludolara/DStruct2Design-nightly", cache_dir=cache_dir)
# print(f"Dataset 'ludolara/DStruct2Design-nightly' downloaded to cache directory: datasets/")
# procthor.save_to_disk("datasets/DStruct2Design-nightly")

# procthor = load_from_disk('datasets/DStruct2Design-nightly')
# print(procthor)


