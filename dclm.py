import sys
from pathlib import Path
import argparse
import subprocess, shlex
import uuid
import json
import os
import git

from transformers.utils import logging
sys.path.append(str(Path(__file__).resolve().parent / 'dclm_color_filter_olmo'))
sys.path.append(str(Path(__file__).resolve().parent / 'DCLM'))
sys.path.append(str(Path(__file__).resolve().parent))  # Add the current directory to the sys.path

from dclm_color_filter_olmo.scripts import select_topk 
from DCLM.training.hyperparameters import get_scale_config
from DCLM.training.dataset_reference import DatasetReference


# NOTE Example usage:
# python dclm.py --filtered_dir /n/holyscratch01/sham_lab/dclm/color_filter_data/first_run/prior/chunk_scores.jsonl --selected_dir /n/holyscratch01/sham_lab/dclm/color_filter_data/pipeline_test --k 10 --type chunks --evaluation light
def main():
    parser = argparse.ArgumentParser(description='Select filtered data and train/evaluate in the DCLM framework')
    parser.add_argument('--filtered_dir', type=str, required=True, 
                        help='Path to Color-Filtered dataset')
    parser.add_argument('--selected_dir', type=str, required=True, 
                        help='Path to selected dataset')
    parser.add_argument('--k', type=int, required=True, 
                        help='How many datapoints to choose')
    parser.add_argument('--type', type=str, choices=['chunks', 'documents'], 
                        required=True, help='Whether to choose documents or chunks')
    parser.add_argument('--dclm_scale', type=str, default='411m_1x', 
                        help='Scale at which to train DCLM')
    parser.add_argument('--evaluation', type=str, required=True, default='light', 
                        choices=['light', 'gsm8k', 'heavy_code', 'heacy_ppl', 'heavy', 'math_code', 'medium', 'mmlu_and_lowvar', 'special', 'cot_fix_plus_gpq_triviaqa', 'cot_fix_plus_gpq'], 
                        help='Name of yaml in DCLM/eval that determines evaluation scheme')
    args = parser.parse_args()

    logger = logging.get_logger(__name__)

    # Select data
    logger.info(f'Arguments: {args}')
    logger.info('Selecting data...')
    documents = args.type == 'documents'
    select_topk.select(args.filtered_dir, args.selected_dir, args.k, documents=documents)

    # DCLM training requirements
    if not Path(f'{args.selected_dir}/manifest.jsonl').is_file():
        manifest_args = shlex.split(f'python -m open_lm.utils.make_wds_manifest --data-dir {args.selected_dir}')
        return_code = subprocess.call(manifest_args)
        if return_code != 0:
            logger.error(f'Could not create manifest. Exiting with code {return_code}')
            sys.exit(return_code)

    dclm_dataset_path = Path('./DCLM/exp_data/datasets/tokenized/')
    dclm_dataset_name =  Path(args.selected_dir).stem
    logger.info('Selection complete.')

    dcnlp_commit_hash, dcnlp_diff = get_git_info()
    json_data = {
        "uuid": str(uuid.uuid4()),
        "name": dclm_dataset_name,
        "dataset_url": args.selected_dir,
        "manifest_url": str(Path(args.selected_dir) / 'manifest.jsonl'),
        "sources": [],
        "tokenized": True,
        "num_tokens": count_tokens(str(Path(args.selected_dir) / 'manifest.jsonl')),
        "size": get_local_dir_size(args.selected_dir),
        "dcnlp_commit_hash": dcnlp_commit_hash,
        "dcnlp_diff": dcnlp_diff,
        "sampling_yaml": None
    }

    json_path = str(dclm_dataset_path / dclm_dataset_name) + '.json'
    with open(json_path, 'w') as file:
        json.dump(json_data, file)

    logger.info(f'Json path: {json_path}')
    assert Path(json_path).is_file()


    # Train DCLM 
    logger.info('Training DCLM...')
    dclm_args = shlex.split(
        f'torchrun --nproc-per-node 4 -m DCLM.training.train -- '
        f'--scale {args.dclm_scale} '
        f'--data-config {json_path} '
        f'--logs ./DCLM/logs '
        f'--torchcompile'
    )
    return_code = subprocess.call(dclm_args)
    if return_code != 0:
        logger.error(f'Training failed with return code {return_code}. Aborting the script.')
        sys.exit(return_code)  

    # Eval DCLM
    # Idea is to get the model uuid evaluate needs by grabbing from most recently
    # made file, which should be the model we just trained.
    model_uuid = get_most_recent_uuid('./DCLM/exp_data/models')
    logger.info('Evaluating DCLM...')
    eval_args = shlex.split(
        f"python DCLM/tools/eval_expdb.py "
        f"--num_gpus 4 "
        f"--no_skip "
        f"--output_dir ./DCLM/exp_data/evals "
        f"--eval_yaml ./DCLM/evalarg/{args.evaluation}.yaml "
        f"-f 'uuid={model_uuid}' "
        f"--skip_perplexity"
    )
    return_code = subprocess.call(eval_args)
    if return_code != 0:
        logger.error(f'Evaluation failed with return code {return_code}.')
        sys.exit(return_code)


"""Helper functions"""
def get_local_dir_size(directory):
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(directory):
        for filename in filenames:
            filepath = os.path.join(dirpath, filename)
            total_size += os.path.getsize(filepath)
    return total_size

def count_tokens(manifest_url, seqlen=2049):
    with Path(manifest_url).open("r") as f:
        manifest = [json.loads(line) for line in f]
    num_tokens = sum(int(line["num_sequences"]) for line in manifest) * seqlen
    return num_tokens

def get_git_info():
    repo = git.Repo('./DCLM')
    dcnlp_commit_hash = repo.head.object.hexsha
    dcnlp_diff = repo.git.diff(repo.head.commit.tree)
    return dcnlp_commit_hash, dcnlp_diff

def get_most_recent_uuid(directory):
    files = [f for f in Path(directory).iterdir() if f.is_file()]
    if not files:
        return None
    latest_file = max(files, key=lambda x: x.stat().st_ctime)
    with latest_file.open('r') as file:
        data = json.load(file)
    return data.get('uuid')

if __name__ == '__main__':
    main()