import numpy as np
import json
import os
import tarfile
import gzip
import pathlib
import argparse
import yaml

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
        for line in f:
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
    """Packages list of paths and indices into tar files contaiing 'max_length'
       .json.gz files.

    Args:
        paths_and_indices (list): List like [Path, [index range]]
        out_path (Path): Where to save .tars
    """
    os.makedirs(out_path, exist_ok=True)
    tar_splits = [paths_and_indices[x : x + max_length] 
                  for x in range(0, len(paths_and_indices), max_length)]
    
    tar_count = 0
    for batch in tar_splits:
        tar_path = os.path.join(out_path, f'{tar_count:05d}.tar')
        with tarfile.open(tar_path, 'w') as tar:
            for i, (path, [start, stop]) in enumerate(batch):
                # npy_data = np.lib.format.open_memmap(path)
                # tokens = npy_data[start : stop]
                tokens = _read_chunk_from_memmap(path, start, stop).tolist()

                # Make json.gz out of tokens and add to tar
                json_path = os.path.join(out_path, f'tar_{tar_count:05d}_chunk_{i:05d}.json.gz')
                with gzip.open(json_path, 'wt') as json_file:
                    json.dump(tokens, json_file)
                tar.add(json_path, arcname=os.path.basename(json_path))
                os.remove(json_path)

        tar_count += 1

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
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process some paths.')
    parser.add_argument('config', type=str, help='Path to the YAML configuration file')
    args = parser.parse_args()
    cfg = read_yaml(args.config)

    SCORE_PATH = cfg['score_path']
    OUT_PATH = cfg['out_path']
    select(SCORE_PATH, OUT_PATH, cfg['k'], cfg['documents'])