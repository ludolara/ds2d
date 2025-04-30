import os
from dotenv import load_dotenv 
from huggingface_hub import HfApi, login
from datasets import load_from_disk
from utils.create_example import create_input
# load_dotenv()  
# login(token=os.environ.get("HF_TOKEN_WRITE"))
api = HfApi()
api.create_repo(repo_id="ludolara/rplan_converted_no_doors", repo_type="dataset", private=False, exist_ok=True)
rplan = load_from_disk('datasets/rplan_converted_no_doors')
rplan = rplan.map(lambda sample: {"input": create_input(sample)})
rplan.push_to_hub("ludolara/rplan_converted_no_doors")

