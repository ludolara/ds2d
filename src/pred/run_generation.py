import argparse
from src.pred.floorplan_generator import FloorplanGenerator
from src.pred.floorplan_generator_openai import FloorplanGenerator as FloorplanGeneratorOpenAI

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str, default="models/Llama-3.3-70B-Instruct")
    parser.add_argument("--lora_adapter_path", type=str, default=None)
    parser.add_argument("--dataset_name_or_path", type=str, default="datasets/rplan_converted")
    parser.add_argument("--test_split", type=str, default="test")
    parser.add_argument("--test_range", type=str, default=None, help="Specify range of test examples to use (e.g., '1,101' for the first 100 samples)")
    parser.add_argument("--max_new_tokens", type=int, default=4096)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--feedback_iterations", type=int, default=5)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output_dir", type=str, default="results_feedback/generations/rplan_3_70B/full_prompt", help="Directory to store the generated outputs")
    return parser.parse_args()

def main():
    args = parse_arguments()

    if args.model_name_or_path == "gpt-4o":
        generator = FloorplanGeneratorOpenAI(
            model_name_or_path=args.model_name_or_path,
            lora_adapter_path=args.lora_adapter_path,
            test_split=args.test_split,
            test_range=args.test_range,
            max_new_tokens=args.max_new_tokens,
            batch_size=args.batch_size,
            device=args.device,
            output_dir=args.output_dir
        )
    else:
        generator = FloorplanGenerator(
            model_name_or_path=args.model_name_or_path,
            lora_adapter_path=args.lora_adapter_path,
            dataset_name_or_path=args.dataset_name_or_path,
            test_split=args.test_split,
            test_range=args.test_range,
            max_new_tokens=args.max_new_tokens,
            batch_size=args.batch_size,
            device=args.device,
            output_dir=args.output_dir
        )
    generator.generate_floorplans_with_feedback(feedback_iterations=args.feedback_iterations)

if __name__ == "__main__":
    main()
