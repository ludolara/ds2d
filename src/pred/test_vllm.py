from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

model_path = "/home/l/luislara/links/scratch/ds2d_v2/models/Llama-3.3-70B-Instruct"
lora_adapter_path = "/home/l/luislara/links/scratch/ds2d_v2/output/rplan_3_70B/"

model = LLM(model=model_path, tensor_parallel_size=4, device="cuda", enable_lora=True)
lora = LoRARequest("lora_adapter", 1, lora_adapter_path)
sampling_params = SamplingParams(max_tokens=50)

prompt = "You are a pirate, talk like a pirate. What is your name?"

outputs = model.generate([prompt], sampling_params, lora_request=lora)
print(outputs[0].outputs[0].text)

