from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch

base_model_path = "models/Llama-3.3-70B-Instruct"       
adapter_path = "output/rplan_8_70B_r256_a512_all"            
merged_model_path = "models/r256_a512-Llama-3.3-70B-Instruct"

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
