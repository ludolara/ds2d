import argparse
import os
from src.pred.floorplan_generator import FloorplanGenerator

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str, default="models/Llama-3.3-70B-Instruct")
    parser.add_argument("--lora_adapter_path", type=str, default=None)
    parser.add_argument("--dataset_name_or_path", type=str, default="datasets/rplan_converted")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--test_range", type=str, default=None, help="Specify range of test examples to use (e.g., '1,101' for the first 100 samples)")
    parser.add_argument("--max_new_tokens", type=int, default=4096)
    parser.add_argument("--batch_size", type=int, default=8)

    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output_dir", type=str, default="results_feedback/generations/rplan_3_70B/full_prompt", help="Directory to store the generated outputs")
    parser.add_argument("--use_sampling", action="store_true", help="Whether to use sampling mode")
    return parser.parse_args()

def main():
    args = parse_arguments()
    
    use_sampling = args.use_sampling
    if use_sampling:
        os.environ["VLLM_USE_V1"] = "0"
    else:
        os.environ["VLLM_USE_V1"] = "1"

    generator = FloorplanGenerator(
        model_name_or_path=args.model_name_or_path,
        lora_adapter_path=args.lora_adapter_path,
        dataset_name_or_path=args.dataset_name_or_path,
        test_split=args.split,
        test_range=args.test_range,
        max_new_tokens=args.max_new_tokens,
        batch_size=args.batch_size,
        device=args.device,
        output_dir=args.output_dir,
        use_sampling=use_sampling
    )
    generator.generate_floorplans()

if __name__ == "__main__":
    main()
