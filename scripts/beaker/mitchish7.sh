#!/usr/bin/env bash
set -exuo pipefail
IFS=$'\n\t'

# Warm HF cache
mkdir -p /root/.cache
pushd /root/.cache
curl "https://storage.googleapis.com/dirkgr-public/huggingface_cache_v2.tar.gz" | tar -xzf -
popd
export HF_DATASETS_OFFLINE=1

torchrun \
  --nnodes ${NUM_NODES}:${NUM_NODES} \
  --nproc-per-node 8 \
  --rdzv_id=101 \
  --rdzv_backend=c10d \
  --rdzv_endpoint=\$BEAKER_LEADER_REPLICA_HOSTNAME:29400 \
  scripts/train.py \
  configs/mitchish7-s3.yaml \
    --run_name=mitchish7 \
    --wandb.name=mitchish7 \
    --model.flash_attention=true \
    --fsdp.wrapping_strategy=by_block_and_size \
    --fsdp.sharding_strategy=SHARD_GRAD_OP \
    --save_folder=runs/ \
    --device_train_microbatch_size=2 \
    --global_train_batch_size=1024 \
    --save_overwrite