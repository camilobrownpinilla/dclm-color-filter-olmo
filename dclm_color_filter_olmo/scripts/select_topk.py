import numpy as np
import torch
import json
import os
import tarfile
import gzip
from pathlib import Path
import shutil
import argparse
import yaml
import multiprocessing
from tqdm import tqdm, trange
import sys

from olmo.util import get_bytes_range
CPU_COUNT = multiprocessing.cpu_count()


def sample_n_tokens(score_path, out_path, n, T, n_processes=CPU_COUNT):
    """Samples tokens according to softmax distribution over scores. 
       Adjust temperature with parameter `t`

    Args:
        score_path (Path): Path to jsonl with scores.
        out_path (Path): Where to write out selected data
        n (int): Number of tokens to select
        T (float): 'Temperature' of softmax distribution
        n_processes (int, optional): How many cpu cores to use. Defaults to CPU_COUNT.
    """
    global dist, scores, softmax_dist, selected_tokens

    # Read in json, store {score: metadata} dict
    dist = {}
    with open(score_path, 'r') as score_file:
        for line in tqdm(score_file, desc='Reading scores', colour='yellow', total=int(35e9 / 512)):
            data = json.loads(line)
            # dist[(str(round(data['score'][0], 6)))] = data['metadata']
            dist[f'{(data["score"][0]):.6f}'] = data['metadata']
            break
            
    # Softmax over scores & sample in parallel
    scores = torch.tensor([float(key) for key in dist.keys()])
    softmax_dist = torch.softmax(scores/T, dim=0).numpy()
    softmax_dist /= softmax_dist.sum() # Explicit normalization b/c of fp precision errors
    selected_tokens = []
    with multiprocessing.Pool(processes=n_processes) as pool:
        with tqdm(desc='Sampling', unit='Chunks', colour='green', total=n//int(1e7)//chunksize) as pbar:
            chunksize = 16
            for result in pool.imap_unordered(_sample, range(n//int(1e7)//chunksize), chunksize=chunksize):
                selected_tokens.append(result)
                pbar.update(1)

    print('\U0001F608 \033[33mTarifying...\033[0m')
    tarify_parallel(selected_tokens, out_path, n_processes=n_processes)


def select_top_n_tokens(score_path, out_path, n, r, n_processes=CPU_COUNT):
    """Selects top n tokens. Saves as tar and copies files r times.

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
    """Packages list of paths and indices into tar files containing 'max_length'
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
                    n_processes=CPU_COUNT):
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
    tar_splits = [paths_and_indices[x : x + max_length * 4]
                  for x in range(0, len(paths_and_indices), max_length * 4)]

    # Prepare arguments for multiprocessing
    args = [
        (batch, out_path, tar_count, max_length)
        for tar_count, batch in enumerate(tar_splits)
    ]

    with multiprocessing.Pool(processes=n_processes) as pool:
        with tqdm(desc='Tarifying', colour='green', total=len(tar_splits)) as pbar:
            for _ in pool.imap_unordered(_process_tar_batch, args):
                pbar.update(1) 


def _sample(_):
        sampled_score = np.random.choice(scores, size=int(1e7), p=softmax_dist)
        tokens = [[dist[f'{score:.6f}']['path'], dist[f'{score:.6f}']['memmap_idx_range']] for score in sampled_score]
        return tokens

def _process_tar_batch(args):
    batch, out_path, tar_count, max_length = args
    tar_path = os.path.join(out_path, f'{tar_count:05d}.tar')
    with tarfile.open(tar_path, 'w') as tar:
        tokens = [] 
        for i, (path, [start, stop]) in enumerate(batch):
            tokens += _read_chunk_from_memmap(path, start, stop).tolist()
            tokens += [187] # Newline token for allenai/eleuther-ai-gpt-neox-20b-pii-special

            if i % 4 == 0 and i > 0: # Ensure > 2048 tokens (4 chunks of 512)
                json_filename = f'tar_{tar_count:05d}_chunk_{i:05d}.json.gz'
                json_path = os.path.join(out_path, json_filename)
                with gzip.open(json_path, 'wt') as json_file:
                    tokens = tokens[:2049] # Be sure there are only 2049 to avoid issues
                    json.dump(tokens, json_file)
                    tokens = []

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


def _read_yaml(file_path):
    with open(file_path, 'r') as file:
        return yaml.safe_load(file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process some paths.')
    parser.add_argument('config', type=str, help='Path to the YAML configuration file')

    args = parser.parse_args()
    cfg = _read_yaml(args.config)

    SCORE_PATH = cfg['score_path']
    OUT_PATH = cfg['out_path']
    n, mode = int(cfg['n']), cfg['mode']

    if cfg['documents'] is not None:
        os.makedirs(OUT_PATH) # No exist_ok to safeguard overwriting selections
        select(SCORE_PATH, OUT_PATH, cfg['k'], cfg['documents'])
    elif mode[0] not in ['sample', 'repeat']:
        print(f'Mode <\033[31m{mode[0]}\033[0m> not supported. Exiting.')
        exit()

    os.makedirs(OUT_PATH) # No exist_ok to safeguard overwriting selections
    if mode[0] == 'sample':
        sample_n_tokens(SCORE_PATH, OUT_PATH, n, float(mode[1])) # mode[1] is T for sample and r for select
    elif mode[0] == 'repeat':
        select_top_n_tokens(SCORE_PATH, OUT_PATH, n, int(mode[1]))