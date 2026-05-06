import json
import os
import glob
import sys
import argparse

def merge_json_files(directory, model_id):
    # Find all JSON files matching the pattern
    pattern = os.path.join(directory, f"captions_{model_id}_rank*.json")
    json_files = glob.glob(pattern)
    
    if not json_files:
        print("No JSON files matching 'captions_rank*.json' found in the directory.")
        return
    
    print(f"Found {len(json_files)} JSON files to merge.")
    
    # Initialize an empty dictionary for the merged result
    merged = {}
    
    # Loop through each JSON file
    for file in json_files:
        print(f"Processing {file}...")
        with open(file, 'r') as f:
            rank_data = json.load(f)
        
        # Process each instance_id in the rank's data
        for instance_id, image_captions in rank_data.items():
            # If instance_id already exists in merged, update its inner dictionary
            if instance_id in merged:
                merged[instance_id].update(image_captions)
            else:
                # Otherwise, add the new instance_id and its image captions
                merged[instance_id] = image_captions
    
    # Define output path and save the merged dictionary
    output_path = os.path.join(directory, f"captions_{model_id}.json")
    
    # Count some statistics for verification
    instance_count = len(merged)
    image_count = sum(len(captions) for captions in merged.values())
    
    with open(output_path, 'w', encoding="utf-8") as f:
        json.dump(merged, f, indent=4, ensure_ascii=False)
    
    print(f"Merged JSON saved to {output_path}")
    print(f"Total instances: {instance_count}")
    print(f"Total images: {image_count}")

if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description="Merge JSON files with captions.")
    argparser.add_argument("--dir", type=str, help="Directory containing JSON files to merge.")
    argparser.add_argument("--model_id", type=str, default="", help="Model identifier for the output file.")
    args = argparser.parse_args()
    
    if not os.path.isdir(args.dir):
        print(f"Error: {args.dir} is not a valid directory.")
        sys.exit(1)

    merge_json_files(args.dir, args.model_id)