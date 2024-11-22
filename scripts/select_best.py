import json
import numpy as np
import os
from pathlib import Path


def load_json_lines(file_path):
    with open(file_path, 'r') as f:
        lines = [json.loads(line) for line in f]
    return lines


def calculate_score_differences(prior_entries, conditional_entries):
    score_diffs = []
    
    for prior, conditional in zip(prior_entries, conditional_entries):
        diff = prior['score'][0] - conditional['score'][0]
        score_diffs.append({
            "score_diff": diff,
            "metadata": prior["metadata"]
        })
    
    return score_diffs


def extract_top_x_percent(entries, percentage=0.1):
    """Get top x% of entries with the lowest score diffs"""
    sorted_entries = sorted(entries, key=lambda x: x['score_diff'])
    top_n = int(len(sorted_entries) * percentage)
    return sorted_entries[:top_n]

def save_top_memmap_segments(top_entries, output_dir):
    """Save the top entries from the memmap segments into a new dir"""
    os.makedirs(output_dir, exist_ok=True)
    memmap_cache = {}

    for entry in top_entries:
        path = entry['metadata']['path']
        idx_range = entry['metadata']['memmap_idx_range']
        start_idx, end_idx = idx_range
        
        if path not in memmap_cache:
            memmap_cache[path] = np.load(path, mmap_mode='r')
        
        segment = memmap_cache[path][start_idx:end_idx]
        output_path = Path(output_dir) / f"top_{Path(path).stem}.npy"
        
        if not output_path.exists():
            np.save(output_path, segment)
        else:
            with open(output_path, 'ab') as f:
                np.save(f, segment)


def main(prior_file, conditional_file, output_dir, top_percentage=0.1):
    prior_entries = load_json_lines(prior_file)
    conditional_entries = load_json_lines(conditional_file)
    score_diffs = calculate_score_differences(prior_entries, conditional_entries)
    top_entries = extract_top_x_percent(score_diffs, percentage=top_percentage)
    
    save_top_memmap_segments(top_entries, output_dir)

if __name__ == "__main__":
    prior_file = 'ckpts/None_1/chunk_scores.jsonl'  # prior scores file
    conditional_file = 'conditional_scores.jsonl'  #  conditional scores file
    output_dir = None  
    top_percentage = 0.1 

    main(prior_file, conditional_file, output_dir, top_percentage)
