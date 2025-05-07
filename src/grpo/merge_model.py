from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch

base_model_path = "models/Llama-3.1-8B-Instruct"       
adapter_path = "output/OLD_rplan_30_8B_non_BD"            
merged_model_path = "models/ds2d-Llama-3.1-8B-Instruct"

base_model = AutoModelForCausalLM.from_pretrained(
    base_model_path,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

tokenizer = AutoTokenizer.from_pretrained(base_model_path)

model = PeftModel.from_pretrained(base_model, adapter_path)

model = model.merge_and_unload()

model.save_pretrained(merged_model_path)
tokenizer.save_pretrained(merged_model_path)
