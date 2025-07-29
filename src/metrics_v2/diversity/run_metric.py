import json
import sys
from pathlib import Path
import torch
from datetime import datetime
from tqdm import tqdm
from pytorch_fid.fid_score import calculate_fid_given_paths
from src.plot.housediffusion_visualizer import HouseDiffusionVisualizerDS2D
from src.plot.direct_visualizer import DirectVisualizer

class DiversityMetricGenerator:
    def __init__(self, results_dir="results_GRPO_70B", resolution=256):
        """
        Initialize the diversity metric generator.
        
        Args:
            results_dir (str): Name of the results directory to process
            resolution (int): Resolution for generated images
        """
        self.results_dir = results_dir
        self.resolution = resolution
        self.visualizer = HouseDiffusionVisualizerDS2D(resolution=resolution)
        self.direct_visualizer = DirectVisualizer(resolution=resolution)
        
        # Set up paths - save in project root under results_diversity
        if Path(results_dir).exists():
            # Running from project root
            self.source_path = Path(f"{results_dir}/generations/rplan_8_70B/full_prompt")
        else:
            # Running from elsewhere
            self.source_path = Path(f"../../../{results_dir}/generations/rplan_8_70B/full_prompt")
            
        # Save results in project root under final_results with organized folders
        self.diversity_base = Path("final_results") / results_dir
        
        # HouseDiffusion visualizations
        self.housediffusion_base = self.diversity_base / "housediffusion_viz"
        self.generated_path = self.housediffusion_base / "generated"
        self.generated_svg_path = self.housediffusion_base / "generated_svg"
        self.ground_truth_path = self.housediffusion_base / "ground_truth"
        
        # Direct visualizations
        self.direct_viz_base = self.diversity_base / "direct_viz"
        self.direct_generated_path = self.direct_viz_base / "generated"
        self.direct_ground_truth_path = self.direct_viz_base / "ground_truth"
        
        # Create directories if they don't exist
        self.generated_path.mkdir(parents=True, exist_ok=True)
        self.generated_svg_path.mkdir(parents=True, exist_ok=True)
        self.ground_truth_path.mkdir(parents=True, exist_ok=True)
        self.direct_generated_path.mkdir(parents=True, exist_ok=True)
        self.direct_ground_truth_path.mkdir(parents=True, exist_ok=True)
        
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
            # Load the generated floorplan data
            with open(generated_json, 'r') as f:
                generated_data = json.load(f)
            
            # Generate HouseDiffusion PNG image for the generated floorplan
            output_path = self.generated_path / f"{sample_id}.png"
            img = self.visualizer.visualize_floorplan_ds2d(
                str(generated_json),
                save_path=str(output_path),
                save_svg=False,
                # show_edges=False
            )
            
            # Generate HouseDiffusion SVG image for the generated floorplan
            svg_output_path = self.generated_svg_path / f"{sample_id}.svg"
            svg_img = self.visualizer.visualize_floorplan_ds2d(
                str(generated_json),
                save_path=str(svg_output_path),
                save_svg=True,
                # show_edges=False
            )
            
            # Generate Direct visualization for the generated floorplan
            self.direct_visualizer.generate_and_save_visualization(generated_data, str(self.direct_generated_path / f"{sample_id}.png"))
                
            # Process ground truth if available in analysis directory
            if analysis_dir.exists():
                ground_truth_json = analysis_dir / "sample.json"
                if ground_truth_json.exists():
                    try:
                        with open(ground_truth_json, 'r') as f:
                            gt_data = json.load(f)
                        
                        # Check if ground truth has the expected format
                        if "spaces" in gt_data:
                            # HouseDiffusion ground truth visualization
                            gt_output_path = self.ground_truth_path / f"{sample_id}.png"
                            gt_img = self.visualizer.visualize_floorplan_ds2d(
                                str(ground_truth_json),
                                save_path=str(gt_output_path),
                                save_svg=False,
                                show_edges=False
                            )
                            
                            # Direct ground truth visualization
                            self.direct_visualizer.generate_and_save_visualization(gt_data, str(self.direct_ground_truth_path / f"{sample_id}.png"))
                        else:
                            print(f"Warning: Ground truth data for sample {sample_id} doesn't have 'spaces' key")
                        
                    except Exception as e:
                        print(f"Warning: Could not process ground truth for sample {sample_id}: {e}")
                else:
                    print(f"Warning: No ground truth file found for sample {sample_id}")
                        
        except Exception as e:
            print(f"❌ Error processing sample {sample_id}: {e}")
    

    def generate_diversity_metrics(self, max_samples=None):
        """
        Generate diversity metrics by processing all available samples.
        
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
        print(f"📁 HouseDiffusion visualizations:")
        print(f"   - Generated images: {self.generated_path}")
        print(f"   - Generated SVG files: {self.generated_svg_path}")
        print(f"   - Ground truth images: {self.ground_truth_path}")
        print(f"📁 Direct visualizations:")
        print(f"   - Generated images: {self.direct_generated_path}")
        print(f"   - Ground truth images: {self.direct_ground_truth_path}")
        
        # Process each sample
        processed = 0
        for sample_dir in tqdm(sample_dirs, desc="Processing samples", unit="sample"):
            sample_id = sample_dir.name
            self.process_single_sample(sample_dir, sample_id)
            processed += 1
        
        print(f"\n🎉 Diversity metric generation complete!")
        print(f"📊 All visualizations saved in: {self.diversity_base}")
        
        # Print summary statistics
        hd_generated_count = len(list(self.generated_path.glob("*.png")))
        hd_svg_count = len(list(self.generated_svg_path.glob("*.svg")))
        hd_gt_count = len(list(self.ground_truth_path.glob("*.png")))
        direct_generated_count = len(list(self.direct_generated_path.glob("*.png")))
        direct_gt_count = len(list(self.direct_ground_truth_path.glob("*.png")))
        
        print(f"📈 Summary:")
        print(f"   HouseDiffusion Visualizations:")
        print(f"     - Generated (PNG): {hd_generated_count}")
        print(f"     - Generated (SVG): {hd_svg_count}")
        print(f"     - Ground truth: {hd_gt_count}")
        print(f"   Direct Visualizations:")
        print(f"     - Generated: {direct_generated_count}")
        print(f"     - Ground truth: {direct_gt_count}")
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

def save_diversity_results(generator, fid_scores, generated_counts, gt_counts, processed_count):
    """
    Save diversity metric results to a JSON file.
    
    Args:
        generator (DiversityMetricGenerator): The generator instance with paths
        fid_scores (dict): The computed FID scores for different visualization types
        generated_counts (dict): Number of generated images per visualization type
        gt_counts (dict): Number of ground truth images per visualization type
        processed_count (int): Number of samples processed
    """
    # Get SVG count
    svg_count = len(list(generator.generated_svg_path.glob("*.svg")))
    
    results = {
        "timestamp": datetime.now().isoformat(),
        "fid_scores": fid_scores,
        "results_directory": generator.results_dir,
        "resolution": generator.resolution,
        "statistics": {
            "housediffusion": {
                "generated_images": generated_counts.get("housediffusion", 0),
                "generated_svg_files": svg_count,
                "ground_truth_images": gt_counts.get("housediffusion", 0)
            },
            "direct": {
                "generated_images": generated_counts.get("direct", 0),
                "ground_truth_images": gt_counts.get("direct", 0)
            },
            "samples_processed": processed_count
        },
        "paths": {
            "housediffusion": {
                "generated_images": str(generator.generated_path),
                "generated_svg_files": str(generator.generated_svg_path),
                "ground_truth_images": str(generator.ground_truth_path)
            },
            "direct": {
                "generated_images": str(generator.direct_generated_path),
                "ground_truth_images": str(generator.direct_ground_truth_path)
            },
            "source_data": str(generator.source_path)
        }
    }
    
    results_file = generator.diversity_base / "diversity.json"
    
    try:
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"💾 Results saved to: {results_file}")
        return True
    except Exception as e:
        print(f"❌ Error saving results to JSON: {e}")
        return False

def main():
    """Main function to run the diversity metric generation."""
    if len(sys.argv) < 2:
        print("Usage: python run_metric.py <results_directory>")
        sys.exit(1)

    results_dir = sys.argv[1]
    generator = DiversityMetricGenerator(
        results_dir=results_dir,
        resolution=512
    )
    
    generator.generate_diversity_metrics() 
    
    # Get counts for both visualization types
    hd_generated_count = len(list(generator.generated_path.glob("*.png")))
    hd_gt_count = len(list(generator.ground_truth_path.glob("*.png")))
    direct_generated_count = len(list(generator.direct_generated_path.glob("*.png")))
    direct_gt_count = len(list(generator.direct_ground_truth_path.glob("*.png")))
    
    # Compute FID scores for both visualization types
    print(f"\n🔍 Computing FID scores for both visualization types...")
    
    hd_fid_score = compute_fid_score(
        generated_path=generator.generated_path,
        ground_truth_path=generator.ground_truth_path
    )
    
    direct_fid_score = compute_fid_score(
        generated_path=generator.direct_generated_path,
        ground_truth_path=generator.direct_ground_truth_path
    )
    
    fid_scores = {
        "housediffusion": hd_fid_score,
        "direct": direct_fid_score
    }
    
    generated_counts = {
        "housediffusion": hd_generated_count,
        "direct": direct_generated_count
    }
    
    gt_counts = {
        "housediffusion": hd_gt_count,
        "direct": direct_gt_count
    }
    
    # Print results
    print(f"\n🎯 Final FID Scores:")
    if hd_fid_score is not None:
        print(f"   HouseDiffusion: {hd_fid_score:.4f}")
    else:
        print(f"   HouseDiffusion: Could not compute")
        
    if direct_fid_score is not None:
        print(f"   Direct: {direct_fid_score:.4f}")
    else:
        print(f"   Direct: Could not compute")
        
    print(f"💡 Lower FID scores indicate better image quality and similarity to real data")
    
    processed_count = len([d for d in generator.source_path.iterdir() if d.is_dir() and d.name.isdigit()])
    save_diversity_results(generator, fid_scores, generated_counts, gt_counts, processed_count)

if __name__ == "__main__":
    main()
