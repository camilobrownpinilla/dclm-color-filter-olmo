import numpy as np
import json
import os
import tarfile
import gzip
from pathlib import Path
import shutil
import argparse
import yaml
import multiprocessing
from tqdm import tqdm

from olmo.util import get_bytes_range

def read_yaml(file_path):
    with open(file_path, 'r') as file:
        return yaml.safe_load(file)


def select(score_path, out_path, k, documents=False):
    """Selects either top k token chunks or documents and tarify's them.

    Args:
        score_path (Path): Path to jsonl with scores.
        out_path (Path): Where to write out selected data 
        k (int): Number of chunks/documents to select.
        documents (bool, optional): If true, selects documents instead of chunks. 
                                    Defaults to False.
    """
    top_k = []
    count = 0

    with open(score_path, 'r') as f:
        for line in tqdm(f, desc='Selecting tokens', colour='green'):
            if count >= k:
                break
            line = json.loads(line)
            metadata = line['metadata']
            npy_path = metadata['path']
            idx_range = metadata['memmap_idx_range']

            if documents:
                # Change idx_range to document chunk is from
                meta_metadata_path = npy_path.replace('.npy', '.csv.gz')
                with gzip.open(meta_metadata_path, 'rt') as meta:
                    for line in meta:
                        doc_start, doc_end = line.split(',')[:2]
                        doc_start, doc_end = int(doc_start), int(doc_end)

                        # Check if chunk from this document
                        if  doc_start <= idx_range[0] and doc_end >= idx_range[1]:
                            idx_range = [doc_start, doc_end]

            top_k.append([npy_path, idx_range])
            count += 1

    tarify(top_k, out_path)

                
def tarify(paths_and_indices, out_path, max_length=8192):
    """
    Packages list of paths and indices into tar files containing 'max_length'
    .json.gz files only.

    Args:
        paths_and_indices (list): List like [Path, [index range]]
        out_path (Path): Where to save .tars
        max_length (int): Required token length for saving files.
    """
    os.makedirs(out_path, exist_ok=True)
    tar_splits = [paths_and_indices[x : x + max_length] 
                  for x in range(0, len(paths_and_indices), max_length)]
    
    tar_count = 0
    for batch in tqdm(tar_splits, desc='Tarifying', colour='green'):
        tar_path = os.path.join(out_path, f'{tar_count:05d}.tar')
        with tarfile.open(tar_path, 'w') as tar:
            for i, (path, [start, stop]) in enumerate(batch):
                tokens = _read_chunk_from_memmap(path, start, stop).tolist()
                
                if len(tokens) == max_length:
                    json_path = os.path.join(out_path, f'tar_{tar_count:05d}_chunk_{i:05d}.json.gz')
                    with gzip.open(json_path, 'wt') as json_file:
                        json.dump(tokens, json_file)
                    
                    tar.add(json_path, arcname=os.path.basename(json_path))
                    os.remove(json_path)  # Clean up intermediate file
        tar_count += 1


def tarify_parallel(paths_and_indices, 
                    out_path, 
                    max_length=8192, 
                    n_processes=multiprocessing.cpu_count()):
    """
    Packages list of paths and indices into tar files containing 'max_length'
    .json.gz files only, processing in parallel.

    Args:
        paths_and_indices (list): List of tuples like [(Path, [index range]), ...]
        out_path (str): Where to save .tars
        max_length (int): Required token length for saving files.
        n_processes (int): Number of processes for parallelization.
    """
    print(f'\033[33mTarifying across {n_processes} processes\033[0m')
    os.makedirs(out_path, exist_ok=True)
    tar_splits = [paths_and_indices[x : x + max_length]
                  for x in range(0, len(paths_and_indices), max_length)]

    # Prepare arguments for multiprocessing
    args = [
        (batch, out_path, tar_count, max_length)
        for tar_count, batch in enumerate(tar_splits)
    ]

    with multiprocessing.Pool(processes=n_processes) as pool:
        with tqdm(desc='Tarifying', colour='green', total=len(tar_splits)) as pbar:
            for _ in pool.imap_unordered(_process_tar_batch, args):
                pbar.update(1) 


def _process_tar_batch(args):
    batch, out_path, tar_count, max_length = args
    tar_path = os.path.join(out_path, f'{tar_count:05d}.tar')
    with tarfile.open(tar_path, 'w') as tar:
        for i, (path, [start, stop]) in enumerate(batch):
            tokens = _read_chunk_from_memmap(path, start, stop).tolist()

            json_filename = f'tar_{tar_count:05d}_chunk_{i:05d}.json.gz'
            json_path = os.path.join(out_path, json_filename)
            with gzip.open(json_path, 'wt') as json_file:
                json.dump(tokens, json_file)

            tar.add(json_path, arcname=os.path.basename(json_filename))
            os.remove(json_path)  # Clean up intermediate file


def _read_chunk_from_memmap(path, start, stop, dtype=np.uint16):
        item_size = dtype(0).itemsize
        bytes_start = start * item_size 
        num_bytes = item_size * (stop - start) # stop - start is chunk length
        buffer = get_bytes_range(path, bytes_start, num_bytes)
        array = np.frombuffer(buffer, dtype=dtype)
        if dtype == np.bool_:
            return array
        else:
            return array.astype(np.int_)


def select_top_n_tokens(score_path, out_path, n, r, n_processes=multiprocessing.cpu_count()):
    """Selects top n tokens and copies the tar files r times.

    Args:
        score_path (Path): Path to jsonl with scores.
        out_path (Path): Where to write out selected data.
        n (int): Number of tokens to select.
        r (int): Number of times to repeat the selection by copying tar files.
        n_processes (int, optional): Number of processes for parallelization. Defaults to CPU count.
    """
    print(f"Selecting top {n} tokens...")
    top_tokens = []
    total_tokens = 0

    with open(score_path, 'r') as f:
        for line in f:
            if total_tokens >= n:
                break
            line = json.loads(line)
            metadata = line['metadata']
            npy_path = metadata['path']
            idx_range = metadata['memmap_idx_range']

            chunk_size = idx_range[1] - idx_range[0]
            if total_tokens + chunk_size > n:
                adjusted_range = [idx_range[0], idx_range[0] + (n - total_tokens)]
                top_tokens.append([npy_path, adjusted_range])
                total_tokens += (n - total_tokens)
            else:
                top_tokens.append([npy_path, idx_range])
                total_tokens += chunk_size

    # Perform tarification once
    print("Tarifying selected tokens...")
    tarify_parallel(top_tokens, out_path, n_processes=n_processes)

    # Copy tar files r-1 times with unique repetition suffixes
    if r > 1:
        print(f"Copying tar files {r - 1} additional times...")
        original_tar_files = sorted(Path(out_path).glob('*.tar'))

        for rep in range(1, r):
            for tar_file in original_tar_files:
                base_name = tar_file.stem  # e.g., '0000'
                new_tar_name = tar_file.parent / f"{base_name}_{rep}.tar"
                shutil.copy(str(tar_file), str(new_tar_name))
                print(f"Copied {tar_file.name} to {new_tar_name.name}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process some paths.')
    parser.add_argument('config', type=str, help='Path to the YAML configuration file')

    args = parser.parse_args()
    cfg = read_yaml(args.config)

    SCORE_PATH = cfg['score_path']
    OUT_PATH = cfg['out_path']
    top_n_tokens, repeat = int(cfg['n']), int(cfg['repeat'])
    os.makedirs(OUT_PATH) # No exist_ok to safeguard overwriting selections

    if top_n_tokens and repeat:
        select_top_n_tokens(SCORE_PATH, OUT_PATH, top_n_tokens, repeat)
    else:
        select(SCORE_PATH, OUT_PATH, cfg['k'], cfg['documents'])