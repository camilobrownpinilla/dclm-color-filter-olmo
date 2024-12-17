import tarfile
import os
import gzip
import glob
import json
import numpy as np
import argparse
import yaml
from multiprocessing import Pool, cpu_count

from pathlib import Path
from tqdm import tqdm
from dolma.tokenizer.data_types import TokenizerOutput
from dolma.tokenizer.memmap_writer import MemmapWriter


def read_yaml(file_path):
    with open(file_path, 'r') as file:
        return yaml.safe_load(file)


def format_tar(file, out_path, dtype=np.uint16, max_tokens=512*1024*1024):
    """Format a tar file containing tokenized documents into a numpy memmap array.
       Save to out_path without using a fixed batch size.

    Args:
        file (Path): Path to tar file containing tokenized documents.
        out_path (Path): Path to save memmap array(s) to.
    """
    os.makedirs(out_path, exist_ok=True)
    file_count = 0
    base_name = Path(file).stem
    parent_name = Path(file).parent.stem # Avoid overwriting if given several dirs of tars  
    
    # Initial path and MemmapWriter setup
    current_out_path = Path(out_path) / f"{parent_name}_{base_name}_{file_count:05d}.npy"
    writer = MemmapWriter(str(current_out_path), dtype=dtype, max_tokens=max_tokens).__enter__()

    id = 0
    remaining_tokens = []  # Carry over tokens if a file is full

    with tarfile.open(file, "r") as tar:
        for document in tar.getmembers():
            if document.isfile() and document.name.endswith('.json.gz'):
                with gzip.open(tar.extractfile(document)) as doc:
                    tokens = json.load(doc)
                    tokenized_output = TokenizerOutput.from_tokens(id=str(id), src=os.path.join(file, document.path), loc=0, tokens=tokens)
                    remaining_tokens.append(tokenized_output)  # Start with remaining tokens

                    # Attempt to write all tokens; collect any that don’t fit
                    remaining_tokens = writer.write_many(remaining_tokens, flush=False)
                    id += 1

                    # If we still have remaining tokens, the current file is full; start a new one
                    if remaining_tokens:
                        writer.__exit__()  
                        file_count += 1  
                        current_out_path = Path(out_path) / f"{parent_name}_{base_name}_{file_count:05d}.npy"
                        writer = MemmapWriter(str(current_out_path), dtype=dtype, max_tokens=max_tokens).__enter__()

        # Ensure any remaining tokens are written after the loop
        if remaining_tokens:
            writer.write_many(remaining_tokens, flush=True)

        # Close the final writer
        writer.__exit__()


def process_tar_file(args):
    """Worker function to process a single tar file."""
    tar_path, memmap_path = args
    try:
        format_tar(tar_path, memmap_path)
        return (tar_path, True, None)
    except Exception as e:
        return (tar_path, False, str(e))


if __name__ == '__main__':   
    parser = argparse.ArgumentParser(description='Process some paths.')
    parser.add_argument('config', type=str, help='Path to the YAML configuration file')
    args = parser.parse_args()
    cfg = read_yaml(args.config)
    
    TAR_PATHS = glob.glob(cfg['tar_paths'])
    MEMMAP_PATH = cfg['memmap_path']
    
    # Determine number of processes
    n_processes = cfg['processes'] or cpu_count()
    print(f'Processing {len(TAR_PATHS)} tar files using {n_processes} processes')
    
    # Create argument tuples for the worker function
    work_args = [(tar_path, MEMMAP_PATH) for tar_path in TAR_PATHS]
    
    # Process tar files in parallel
    results = []
    with Pool(n_processes) as pool:
        with tqdm(desc='Writing memmaps', colour='green', total=len(TAR_PATHS)) as pbar:
            for result in pool.imap_unordered(process_tar_file, work_args):
                tar_path, success, error = result
                if not success:
                    print(f'\033[31mERROR\033[0m: Failed to process {tar_path}: {error}')
                results.append(result)
                pbar.update(1)
    
    # Report final statistics
    successful = sum(1 for _, success, _ in results if success)
    print(f'\n\033[32mProcessing complete\033[0m: {successful}/{len(TAR_PATHS)} files processed successfully')


