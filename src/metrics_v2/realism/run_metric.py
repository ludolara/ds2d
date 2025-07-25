import json
from pathlib import Path
import torch
from datetime import datetime
from tqdm import tqdm
from pytorch_fid.fid_score import calculate_fid_given_paths
from floorplan_image_generator import HouseDiffusionVisualizerDS2D

class RealismMetricGenerator:
    def __init__(self, results_dir="results_GRPO_70B_ckpt400_sampling", resolution=512):
        """
        Initialize the realism metric generator.
        
        Args:
            results_dir (str): Name of the results directory to process
            resolution (int): Resolution for generated images
        """
        self.results_dir = results_dir
        self.resolution = resolution
        self.visualizer = HouseDiffusionVisualizerDS2D(resolution=resolution)
        
        # Set up paths - save in project root under results_realism
        if Path(results_dir).exists():
            # Running from project root
            self.source_path = Path(f"{results_dir}/generations/rplan_8_70B/full_prompt")
        else:
            # Running from elsewhere
            self.source_path = Path(f"../../../{results_dir}/generations/rplan_8_70B/full_prompt")
            
        # Save results in project root under results_realism
        self.realism_base = Path("results_realism") / results_dir
        self.generated_path = self.realism_base / "generated"
        self.generated_svg_path = self.realism_base / "generated_svg"
        self.ground_truth_path = self.realism_base / "ground_truth"
        
        # Create directories if they don't exist
        self.generated_path.mkdir(parents=True, exist_ok=True)
        self.generated_svg_path.mkdir(parents=True, exist_ok=True)
        self.ground_truth_path.mkdir(parents=True, exist_ok=True)
        
    def process_single_sample(self, sample_dir, sample_id):
        """
        Process a single sample directory and generate visualizations.
        
        Args:
            sample_dir (Path): Path to the sample directory
            sample_id (str): Sample identifier
        """
        generated_json = sample_dir / "0.json"
        analysis_dir = sample_dir / "analysis"
        
        if not generated_json.exists():
            print(f"Warning: {generated_json} not found, skipping sample {sample_id}")
            return
        
        try:
            # Generate PNG image for the generated floorplan
            output_path = self.generated_path / f"{sample_id}.png"
            img = self.visualizer.visualize_floorplan_ds2d(
                str(generated_json),
                save_path=str(output_path),
                save_svg=False,
                show_edges=False
            )
            
            # Generate SVG image for the generated floorplan
            svg_output_path = self.generated_svg_path / f"{sample_id}.svg"
            svg_img = self.visualizer.visualize_floorplan_ds2d(
                str(generated_json),
                save_path=str(svg_output_path),
                save_svg=True,
                show_edges=False
            )
            
            # if img is not None:
            #     print(f"✅ Generated visualization for sample {sample_id}")
            # else:
            #     print(f"❌ Failed to generate visualization for sample {sample_id}")
                
            # Process ground truth if available in analysis directory
            if analysis_dir.exists():
                ground_truth_json = analysis_dir / "sample.json"
                if ground_truth_json.exists():
                    try:
                        with open(ground_truth_json, 'r') as f:
                            gt_data = json.load(f)
                        
                        # Check if ground truth has the expected format
                        # The sample.json already has "spaces" at the top level
                        if "spaces" in gt_data:
                            gt_output_path = self.ground_truth_path / f"{sample_id}.png"
                            gt_img = self.visualizer.visualize_floorplan_ds2d(
                                str(ground_truth_json),
                                save_path=str(gt_output_path),
                                save_svg=False,
                                show_edges=False
                            )
                            
                            # if gt_img is not None:
                            #     print(f"✅ Generated ground truth visualization for sample {sample_id}")
                            # else:
                            #     print(f"❌ Failed to generate ground truth visualization for sample {sample_id}")
                        else:
                            print(f"Warning: Ground truth data for sample {sample_id} doesn't have 'spaces' key")
                        
                    except Exception as e:
                        print(f"Warning: Could not process ground truth for sample {sample_id}: {e}")
                else:
                    print(f"Warning: No ground truth file found for sample {sample_id}")
                        
        except Exception as e:
            print(f"❌ Error processing sample {sample_id}: {e}")
    
    def generate_realism_metrics(self, max_samples=None):
        """
        Generate realism metrics by processing all available samples.
        
        Args:
            max_samples (int, optional): Maximum number of samples to process
        """
        if not self.source_path.exists():
            print(f"❌ Source path {self.source_path} does not exist!")
            return
        
        # Get all sample directories (numbered directories)
        sample_dirs = [d for d in self.source_path.iterdir() if d.is_dir() and d.name.isdigit()]
        sample_dirs.sort(key=lambda x: int(x.name))
        
        if max_samples:
            sample_dirs = sample_dirs[:max_samples]
        
        print(f"🏠 Processing {len(sample_dirs)} samples from {self.source_path}")
        print(f"📁 Saving generated images to: {self.generated_path}")
        print(f"📁 Saving generated SVG files to: {self.generated_svg_path}")
        print(f"📁 Saving ground truth images to: {self.ground_truth_path}")
        
        # Process each sample
        processed = 0
        for sample_dir in tqdm(sample_dirs, desc="Processing samples", unit="sample"):
            sample_id = sample_dir.name
            # print(f"Processing sample {sample_id} ({processed + 1}/{len(sample_dirs)})")
            self.process_single_sample(sample_dir, sample_id)
            processed += 1
        
        print(f"\n🎉 Realism metric generation complete!")
        print(f"📊 Generated images saved in: {self.realism_base}")
        
        # Print summary statistics
        generated_count = len(list(self.generated_path.glob("*.png")))
        svg_count = len(list(self.generated_svg_path.glob("*.svg")))
        gt_count = len(list(self.ground_truth_path.glob("*.png")))
        
        print(f"📈 Summary:")
        print(f"   - Generated visualizations (PNG): {generated_count}")
        print(f"   - Generated visualizations (SVG): {svg_count}")
        print(f"   - Ground truth visualizations: {gt_count}")
        print(f"   - Total samples processed: {processed}")

def compute_fid_score(generated_path, ground_truth_path, device="cuda" if torch.cuda.is_available() else "cpu"):
    """
    Compute FID score between generated and ground truth images using pytorch_fid library.
    
    Args:
        generated_path (Path): Path to generated images
        ground_truth_path (Path): Path to ground truth images
        device (str): Device to use for computation
    
    Returns:
        float: FID score
    """
    print(f"\n🔍 Computing FID score using pytorch_fid library...")
    print(f"📱 Using device: {device}")
    
    # Check if directories exist and contain images
    generated_images = list(generated_path.glob("*.png"))
    gt_images = list(ground_truth_path.glob("*.png"))
    
    if len(generated_images) == 0 or len(gt_images) == 0:
        print("❌ No valid images found in one or both folders!")
        return None
    
    # Check minimum sample sizes for reliable FID computation
    min_samples = 10  # Minimum recommended for FID
    if len(generated_images) < min_samples or len(gt_images) < min_samples:
        print(f"⚠️  Warning: Small sample size detected. For reliable FID scores, use at least {min_samples} images per set.")
        print(f"   Generated: {len(generated_images)}, Ground truth: {len(gt_images)}")
    
    print(f"🖼️  Generated images: {len(generated_images)}")
    print(f"🖼️  Ground truth images: {len(gt_images)}")
    
    try:
        # Use pytorch_fid library to calculate FID score
        print("📊 Calculating FID score...")
        fid_score = calculate_fid_given_paths(
            [str(generated_path), str(ground_truth_path)],
            batch_size=64,  # Process in batches for memory efficiency
            device=device,
            dims=2048,  # Inception v3 feature dimension
            # num_workers=0  # Set to 0 to avoid multiprocessing issues
        )
        
        print(f"📊 FID Score: {fid_score:.4f}")
        return fid_score
        
    except Exception as e:
        print(f"❌ Error computing FID score: {e}")
        return None


def save_realism_results(generator, fid_score, generated_count, gt_count, processed_count):
    """
    Save realism metric results to a JSON file.
    
    Args:
        generator (RealismMetricGenerator): The generator instance with paths
        fid_score (float): The computed FID score
        generated_count (int): Number of generated images
        gt_count (int): Number of ground truth images
        processed_count (int): Number of samples processed
    """
    # Get SVG count
    svg_count = len(list(generator.generated_svg_path.glob("*.svg")))
    
    results = {
        "timestamp": datetime.now().isoformat(),
        "fid_score": fid_score,
        "results_directory": generator.results_dir,
        "resolution": generator.resolution,
        "statistics": {
            "generated_images": generated_count,
            "generated_svg_files": svg_count,
            "ground_truth_images": gt_count,
            "samples_processed": processed_count
        },
        "paths": {
            "generated_images": str(generator.generated_path),
            "generated_svg_files": str(generator.generated_svg_path),
            "ground_truth_images": str(generator.ground_truth_path),
            "source_data": str(generator.source_path)
        }
    }
    
    results_file = generator.realism_base / "realism.json"
    
    try:
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"💾 Results saved to: {results_file}")
        return True
    except Exception as e:
        print(f"❌ Error saving results to JSON: {e}")
        return False


def main():
    """Main function to run the realism metric generation."""
    generator = RealismMetricGenerator(
        results_dir="results_GRPO_70B_ckpt400_sampling",
        resolution=256
    )
    
    generator.generate_realism_metrics() 
    
    generated_count = len(list(generator.generated_path.glob("*.png")))
    gt_count = len(list(generator.ground_truth_path.glob("*.png")))
    
    fid_score = compute_fid_score(
        generated_path=generator.generated_path,
        ground_truth_path=generator.ground_truth_path
    )
    
    if fid_score is not None:
        print(f"\n🎯 Final FID Score: {fid_score:.4f}")
        print(f"💡 Lower FID scores indicate better image quality and similarity to real data")
    else:
        print("\n❌ Could not compute FID score")
    
    processed_count = len([d for d in generator.source_path.iterdir() if d.is_dir() and d.name.isdigit()])
    save_realism_results(generator, fid_score, generated_count, gt_count, processed_count)

if __name__ == "__main__":
    main()
