import json
import sys
import os
import argparse
import yaml


def read_yaml(file_path):
    with open(file_path, 'r') as file:
        return yaml.safe_load(file)

def load_jsonl(file_path):
    """Loads a JSONL file into a list of dictionaries."""
    with open(file_path, 'r') as f:
        return [json.loads(line) for line in f]

def save_jsonl(data, output_path):
    """Saves a list of dictionaries to a JSONL file."""
    with open(output_path, 'w') as f:
        for item in data:
            f.write(json.dumps(item) + '\n')

def load_folder_jsonl(folder_path):
    """Loads all JSONL files in a folder into a single list of dictionaries."""
    combined_data = []
    for file_name in os.listdir(folder_path):
        if file_name.endswith('.jsonl') and file_name.startswith('chunk_scores'):
            file_path = os.path.join(folder_path, file_name)
            combined_data.extend(load_jsonl(file_path))
    return combined_data

def subtract_scores(folder1_data, folder2_data):
    """Subtracts scores from aligned entries in two lists of dictionaries."""
    result = []
    # Create a lookup dictionary for quick access to entries by `memmap_idx_range`
    file2_dict = {tuple(entry['metadata']['memmap_idx_range']): entry for entry in folder2_data}

    for entry1 in folder1_data:
        memmap_idx_range = tuple(entry1['metadata']['memmap_idx_range'])
        entry2 = file2_dict.get(memmap_idx_range)

        if entry2:
            # Calculate the score difference
            score_diff = entry1['score'][0] - entry2['score'][0]
            result_entry = {
                "score": [score_diff],
                "metadata": entry1["metadata"]
            }
            result.append(result_entry)

    # Sort the results by score in ascending order
    result = sorted(result, key=lambda x: x["score"][0])
    return result

def main(folder1_path, folder2_path, output_path):
    # Load data from both JSONL files

    folder1_data = load_folder_jsonl(folder1_path)
    folder2_data = load_folder_jsonl(folder2_path)

    # Calculate score differences and sort
    result = subtract_scores(folder1_data, folder2_data)

    # Save the result to a new JSONL file
    save_jsonl(result, output_path)
    print(f"Processed data saved to {output_path}")

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Process some paths.')
    parser.add_argument('config', type=str, help='Path to the YAML configuration file')
    args = parser.parse_args()
    cfg = read_yaml(args.config)
    
    folder1_path = os.path.join(cfg.checkpoints_path, cfg.prior_path)
    folder2_path = os.path.join(cfg.checkpoints_path, cfg.conditional_path)
    output_path = os.path.join(cfg.checkpoints_path, "combined", "chunk_scores.jsonl")

    main(folder1_path, folder2_path, output_path)