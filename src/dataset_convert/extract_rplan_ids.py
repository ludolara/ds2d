#!/usr/bin/env python3
"""
Script to extract rplan_id values from all analysis/sample.json files 
in the results8_70B folder and save them to list.txt
"""

import os
import json
import glob
from pathlib import Path

def extract_rplan_ids(base_folder="results8_70B", output_file="list.txt"):
    """
    Extract rplan_id from all analysis/sample.json files in the base folder
    
    Args:
        base_folder (str): The folder to search in
        output_file (str): Output file to save the list of rplan_ids
    """
    rplan_ids = []
    
    # Pattern to find all analysis/sample.json files
    pattern = os.path.join(base_folder, "generations", "rplan_8_70B", "full_prompt", "*", "analysis", "sample.json")
    
    print(f"Searching for files matching pattern: {pattern}")
    
    # Find all matching files
    json_files = glob.glob(pattern)
    
    if not json_files:
        print(f"No files found matching pattern: {pattern}")
        print(f"Make sure the {base_folder} folder exists and contains subdirectories with analysis/sample.json files")
        return
    
    print(f"Found {len(json_files)} analysis/sample.json files")
    
    # Process each JSON file
    for json_file in sorted(json_files):
        try:
            print(f"Processing: {json_file}")
            
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Extract rplan_id
            rplan_id = data.get('rplan_id')
            
            if rplan_id:
                rplan_ids.append(rplan_id)
                print(f"  Found rplan_id: {rplan_id}")
            else:
                print(f"  Warning: No rplan_id found in {json_file}")
                
        except json.JSONDecodeError as e:
            print(f"  Error: Failed to parse JSON in {json_file}: {e}")
        except FileNotFoundError:
            print(f"  Error: File not found: {json_file}")
        except Exception as e:
            print(f"  Error: Unexpected error processing {json_file}: {e}")
    
    # Save to output file
    if rplan_ids:
        # Remove duplicates while preserving order
        unique_rplan_ids = list(dict.fromkeys(rplan_ids))
        
        with open(output_file, 'w', encoding='utf-8') as f:
            for rplan_id in unique_rplan_ids:
                f.write(f"{rplan_id}.json\n")
        
        print(f"\nSuccessfully extracted {len(unique_rplan_ids)} unique rplan_ids")
        print(f"Saved to: {output_file}")
        
        # Display first few entries as preview
        print(f"\nPreview of extracted rplan_ids:")
        for i, rplan_id in enumerate(unique_rplan_ids[:10]):
            print(f"  {rplan_id}")
        if len(unique_rplan_ids) > 10:
            print(f"  ... and {len(unique_rplan_ids) - 10} more")
            
    else:
        print("No rplan_ids found in any of the files")

def main():
    """Main function"""
    # Check if results8_70B folder exists
    base_folder = "results8_70B"
    
    if not os.path.exists(base_folder):
        print(f"Error: Folder '{base_folder}' does not exist in current directory")
        print(f"Current directory: {os.getcwd()}")
        print("Please make sure you're running this script from the correct location")
        return
    
    print(f"Starting extraction from {base_folder} folder...")
    extract_rplan_ids(base_folder)

if __name__ == "__main__":
    main() 