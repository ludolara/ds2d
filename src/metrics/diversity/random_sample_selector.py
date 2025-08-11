#!/usr/bin/env python3
"""
Random Sample Selector Script

This script randomly selects 20 samples from both direct_viz and housediffusion_viz
directories and copies them to organized subfolders for diversity analysis.
"""

import os
import random
import shutil
from pathlib import Path

def get_png_files(directory):
    """Get all PNG files from a directory."""
    directory = Path(directory)
    if not directory.exists():
        print(f"Warning: Directory {directory} does not exist!")
        return []
    
    png_files = list(directory.glob("*.png"))
    return png_files

def create_output_directory(base_name="random_samples"):
    """Create a unique output directory with subfolders."""
    counter = 1
    output_dir = Path(base_name)
    
    while output_dir.exists():
        output_dir = Path(f"{base_name}_{counter}")
        counter += 1
    
    output_dir.mkdir(parents=True)
    
    # Create subfolders with generated and ground_truth subdirectories
    hd_viz_dir = output_dir / "hd_viz"
    direct_viz_dir = output_dir / "direct_viz"
    
    # Create the main visualization type folders
    hd_viz_dir.mkdir(exist_ok=True)
    direct_viz_dir.mkdir(exist_ok=True)
    
    # Create generated and ground_truth subfolders
    hd_viz_generated = hd_viz_dir / "generated"
    hd_viz_ground_truth = hd_viz_dir / "ground_truth"
    direct_viz_generated = direct_viz_dir / "generated"
    direct_viz_ground_truth = direct_viz_dir / "ground_truth"
    
    hd_viz_generated.mkdir(exist_ok=True)
    hd_viz_ground_truth.mkdir(exist_ok=True)
    direct_viz_generated.mkdir(exist_ok=True)
    direct_viz_ground_truth.mkdir(exist_ok=True)
    
    return output_dir, hd_viz_generated, hd_viz_ground_truth, direct_viz_generated, direct_viz_ground_truth

def copy_samples(files, generated_output_dir, ground_truth_output_dir, ground_truth_source_dir, prefix, num_samples=20):
    """Copy random samples to output directory with proper naming."""
    if len(files) < num_samples:
        print(f"Warning: Only {len(files)} files available, but {num_samples} requested!")
        num_samples = len(files)
    
    selected_files = random.sample(files, num_samples)
    
    copied_files = []
    ground_truth_copied = []
    
    for i, file_path in enumerate(selected_files, 1):
        # Create new filename with index (keeping original name)
        new_filename = f"{i:02d}_{file_path.name}"
        
        # Copy generated sample
        dest_path = generated_output_dir / new_filename
        try:
            shutil.copy2(file_path, dest_path)
            copied_files.append((file_path.name, new_filename))
            print(f"Copied generated: {file_path.name} -> {new_filename}")
        except Exception as e:
            print(f"Error copying generated {file_path}: {e}")
        
        # Copy corresponding ground truth sample
        ground_truth_source_file = ground_truth_source_dir / file_path.name
        if ground_truth_source_file.exists():
            ground_truth_dest_path = ground_truth_output_dir / new_filename
            try:
                shutil.copy2(ground_truth_source_file, ground_truth_dest_path)
                ground_truth_copied.append((file_path.name, new_filename))
                print(f"Copied ground truth: {file_path.name} -> {new_filename}")
            except Exception as e:
                print(f"Error copying ground truth {ground_truth_source_file}: {e}")
        else:
            print(f"Warning: No ground truth found for {file_path.name}")
    
    return copied_files, ground_truth_copied

def main():
    """Main function to run the random sampling."""
    print("Random Sample Selector for Diversity Analysis")
    print("=" * 50)
    
    # Set random seed for reproducibility (optional)
    # random.seed(42)
    
    # Define source directories
    base_dir = Path("final_results/results8_GRPO_70B")
    direct_viz_generated_dir = base_dir / "direct_viz" / "generated"
    direct_viz_ground_truth_dir = base_dir / "direct_viz" / "ground_truth"
    housediffusion_viz_generated_dir = base_dir / "housediffusion_viz" / "generated"
    housediffusion_viz_ground_truth_dir = base_dir / "housediffusion_viz" / "ground_truth"
    
    # Get all PNG files from both directories
    print(f"Scanning {direct_viz_generated_dir}...")
    direct_viz_files = get_png_files(direct_viz_generated_dir)
    print(f"Found {len(direct_viz_files)} PNG files in direct_viz")
    
    print(f"Scanning {housediffusion_viz_generated_dir}...")
    housediffusion_viz_files = get_png_files(housediffusion_viz_generated_dir)
    print(f"Found {len(housediffusion_viz_files)} PNG files in housediffusion_viz")
    
    if not direct_viz_files and not housediffusion_viz_files:
        print("No PNG files found in either directory!")
        return
    
    # Create output directory with subfolders
    output_dir, hd_viz_generated, hd_viz_ground_truth, direct_viz_generated, direct_viz_ground_truth = create_output_directory("random_samples")
    print(f"\nCreated output directory: {output_dir}")
    print(f"  - HouseDiffusion samples: {hd_viz_generated}")
    print(f"  - HouseDiffusion ground truth: {hd_viz_ground_truth}")
    print(f"  - Direct visualization samples: {direct_viz_generated}")
    print(f"  - Direct visualization ground truth: {direct_viz_ground_truth}")
    
    # Copy random samples from each directory to respective subfolders
    total_copied = 0
    total_ground_truth_copied = 0
    
    if direct_viz_files:
        print(f"\nSelecting 20 random samples from direct_viz...")
        direct_copied, direct_gt_copied = copy_samples(
            direct_viz_files, 
            direct_viz_generated, 
            direct_viz_ground_truth, 
            direct_viz_ground_truth_dir,
            "direct", 
            20
        )
        total_copied += len(direct_copied)
        total_ground_truth_copied += len(direct_gt_copied)
    
    if housediffusion_viz_files:
        print(f"\nSelecting 20 random samples from housediffusion_viz...")
        housediff_copied, housediff_gt_copied = copy_samples(
            housediffusion_viz_files, 
            hd_viz_generated, 
            hd_viz_ground_truth, 
            housediffusion_viz_ground_truth_dir,
            "housediff", 
            20
        )
        total_copied += len(housediff_copied)
        total_ground_truth_copied += len(housediff_gt_copied)
    
    print(f"\n" + "=" * 50)
    print(f"Summary:")
    print(f"Total generated files copied: {total_copied}")
    print(f"Total ground truth files copied: {total_ground_truth_copied}")
    print(f"Output directory: {output_dir.absolute()}")
    print(f"Random seed used: 42 (for reproducibility)")
    
    # Create a summary file
    summary_file = output_dir / "selection_summary.txt"
    with open(summary_file, 'w') as f:
        f.write("Random Sample Selection Summary\n")
        f.write("=" * 40 + "\n\n")
        f.write(f"Random seed: 42\n")
        f.write(f"Total generated files copied: {total_copied}\n")
        f.write(f"Total ground truth files copied: {total_ground_truth_copied}\n")
        f.write(f"Direct_viz samples: {len(direct_copied) if 'direct_copied' in locals() else 0}\n")
        f.write(f"Direct_viz ground truth: {len(direct_gt_copied) if 'direct_gt_copied' in locals() else 0}\n")
        f.write(f"Housediffusion_viz samples: {len(housediff_copied) if 'housediff_copied' in locals() else 0}\n")
        f.write(f"Housediffusion_viz ground truth: {len(housediff_gt_copied) if 'housediff_gt_copied' in locals() else 0}\n\n")
        
        f.write(f"Organization:\n")
        f.write(f"  - HouseDiffusion samples saved to: hd_viz/generated/\n")
        f.write(f"  - HouseDiffusion ground truth saved to: hd_viz/ground_truth/\n")
        f.write(f"  - Direct visualization samples saved to: direct_viz/generated/\n")
        f.write(f"  - Direct visualization ground truth saved to: direct_viz/ground_truth/\n\n")
        
        if 'direct_copied' in locals():
            f.write("Direct_viz selected files (in direct_viz/generated/ subfolder):\n")
            for original, new in direct_copied:
                f.write(f"  {original} -> {new}\n")
            f.write("\n")
        
        if 'direct_gt_copied' in locals():
            f.write("Direct_viz ground truth files (in direct_viz/ground_truth/ subfolder):\n")
            for original, new in direct_gt_copied:
                f.write(f"  {original} -> {new}\n")
            f.write("\n")
        
        if 'housediff_copied' in locals():
            f.write("Housediffusion_viz selected files (in hd_viz/generated/ subfolder):\n")
            for original, new in housediff_copied:
                f.write(f"  {original} -> {new}\n")
            f.write("\n")
        
        if 'housediff_gt_copied' in locals():
            f.write("Housediffusion_viz ground truth files (in hd_viz/ground_truth/ subfolder):\n")
            for original, new in housediff_gt_copied:
                f.write(f"  {original} -> {new}\n")
    
    print(f"Selection summary saved to: {summary_file}")

if __name__ == "__main__":
    main() 