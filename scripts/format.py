import tarfile
import os
import gzip
import glob
import json
import numpy as np

from pathlib import Path
from tqdm import tqdm
from dolma.tokenizer.data_types import TokenizerOutput
from dolma.tokenizer.memmap_writer import MemmapWriter

TEST_TAR = '/n/holyscratch01/sham_lab/dclm/data/dclm-tokenized-test/00000001.tar'
MEMMAP_PATH = '/n/holyscratch01/sham_lab/dclm/color_filter_data/memmap_test'
MEMMAP_DTYPE = np.uint16
MAX_TOKENS = 512 * 1024 * 1024

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
    
    # Initial path and MemmapWriter setup
    current_out_path = Path(out_path) / f"{base_name}_{file_count:05d}.npy"
    writer = MemmapWriter(str(current_out_path), dtype=dtype, max_tokens=max_tokens).__enter__()

    id = 0
    remaining_tokens = []  # Carry over tokens if a file is full

    with tarfile.open(file, "r") as tar:
        for document in tqdm(tar.getmembers(), desc='Writing Memmaps'):
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
                        current_out_path = Path(out_path) / f"{base_name}_{file_count:05d}.npy"
                        writer = MemmapWriter(str(current_out_path), dtype=dtype, max_tokens=max_tokens).__enter__()

        # Ensure any remaining tokens are written after the loop
        if remaining_tokens:
            writer.write_many(remaining_tokens, flush=True)

        # Close the final writer
        writer.__exit__()



if __name__ == '__main__':
    tars = glob.glob('/n/holyscratch01/sham_lab/dclm/data/dclm-tokenized-test/*.tar')
    for tar in tars:
        print(f'Formatting {tar}')
        format_tar(tar, MEMMAP_PATH)
    
                    





    