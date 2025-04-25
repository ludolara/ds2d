from datasets import load_from_disk
from utils.constants import SYSTEM_PROMPT
from utils import create_input, create_output

def get_custom_dataset(dataset_config, tokenizer, split):
    dataset = load_from_disk(dataset_config.data_path)[split]
    dataset = dataset.shuffle(seed=42)

    def process_sample(sample):
        system_prompt = SYSTEM_PROMPT

        prompt = (
            f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n"
            f"{system_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>\n"
            f"{create_input(sample)}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n"
        )
        
        prompt_tokens = tokenizer(prompt, add_special_tokens=False)
        
        response = f"{create_output(sample)}<|eot_id|>"
        response_tokens = tokenizer(response, add_special_tokens=False)
        
        input_ids = prompt_tokens['input_ids'] + response_tokens['input_ids']
        attention_mask = [1] * len(input_ids)
        
        labels = [-100] * len(prompt_tokens['input_ids']) + response_tokens['input_ids']
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels
        }
    
    return dataset.map(process_sample, remove_columns=dataset.column_names)
