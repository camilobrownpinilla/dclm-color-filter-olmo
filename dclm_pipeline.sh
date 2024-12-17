#!/bin/bash

# Usage: ./dclm_pipeline.sh [--skip-training]

SKIP_TRAINING=false

# Check if the --skip-training flag is provided
if [[ "$1" == "--skip-training" ]]; then
    SKIP_TRAINING=true
fi

echo "Starting the pipeline..."

# Step 1: Format
echo "Step 1: Formatting data..."
python scripts/format.py configs/dclm/format.yaml
if [ $? -ne 0 ]; then
    echo "Error in formatting step. Exiting."
    exit 1
fi

export CHECKPOINTS_PATH=/n/netscratch/sham_lab/Everyone/dclm/color_filter/dclm-pipeline-checkpointing-test
# Optional Step 2: Training
if [ "$SKIP_TRAINING" = false ]; then
    echo "Step 2: Training models..."
    sbatch scripts/launch_sweep.sh configs/sweeps/pretrain.yaml 
    sbatch scripts/launch_sweep.sh configs/sweeps/finetune-dclm.yaml 
else
    echo "Skipping training step."
fi

# Step 3: Scoring
echo "Step 3: Scoring..."
sbatch scripts/launch_sweep.sh configs/sweeps/score-dclm.yaml

# Step 4: Combine-Sort
echo "Step 4: Combining and sorting..."
python scripts/combine_sort.py configs/dclm/combine_sort.yaml
if [ $? -ne 0 ]; then
    echo "Error in combine-sort step. Exiting."
    exit 1
fi

# Step 5: select-topk
echo "Step 5: Selecting top k (or documents)..."
python scripts/select_topk.py configs/dclm/select_topk.yaml
if [ $? -ne 0 ]; then
    echo "Error in select-topk step. Exiting."
    exit 1
fi

echo "Pipeline completed successfully"