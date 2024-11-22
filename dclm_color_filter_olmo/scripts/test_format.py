import tarfile
import numpy as np
import json
import gzip
import os
from transformers import AutoTokenizer


TEST_TAR = '/n/holyscratch01/sham_lab/dclm/data/dclm-tokenized-test/00000001.tar'
MEMMAP_PATH = '/n/holyscratch01/sham_lab/dclm/color_filter_data/memmap_test'

def test_format_tar_to_memmap(tar_path, memmap_path, memmap_dtype=np.uint16):
    """
    Test that the indices of the memmap array correspond to the correct files
    in the original tar file.

    Parameters:
        tar_path (str): Path to the tar file containing .json.gz files.
        memmap_path (str): Path to the generated memmap array.
        memmap_dtype (data-type): Expected data type of the memmap array elements.
    """
    tokenizer = AutoTokenizer.from_pretrained('allenai/eleuther-ai-gpt-neox-20b-pii-special')
    memmap = np.memmap(memmap_path, dtype=memmap_dtype, mode='r')

    with tarfile.open(tar_path, 'r') as tar:
        # Iterate over the tar file members and track index
        for index, member in enumerate(tar.getmembers()):
            if member.isfile() and member.name.endswith('.json.gz'):
                with tar.extractfile(member) as file:
                    with gzip.open(file, 'rt') as gz_file:
                        file_data = json.load(gz_file)

                        # Check if memmap at the current index matches the data
                        memmap_data = memmap[index * 2049 : (index + 1) * 2049]

                        # Assert equality
                        assert np.array_equal(memmap_data, file_data), \
                            f"""Mismatch at index {index} for file {member.name}:\n 
                            {memmap_data==file_data}\n 
                            Memmap:\n{tokenizer.decode(memmap_data[:100])}\n 
                            {'*' * 80}\n
                            File:\n{tokenizer.decode(file_data[:100])}"""

    print("Test passed: All memmap indices correspond to the correct tar file contents.")


test_format_tar_to_memmap(TEST_TAR, f"{MEMMAP_PATH}/00000001_00000.npy")