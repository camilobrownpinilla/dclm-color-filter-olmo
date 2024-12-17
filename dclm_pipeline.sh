#!/bin/bash

# Usage: ./dclm_pipeline.sh [--skip-format] [--skip-training] [--skip-scoring] [--skip-combine-sort] [--skip-select-topk]

# Initialize skip flags
SKIP_FORMAT=false
SKIP_TRAINING=false
SKIP_SCORING=false
SKIP_COMBINE_SORT=false
SKIP_SELECT_TOPK=false

# Parse optional flags
for arg in "$@"; do
    case $arg in
        --skip-format)
            SKIP_FORMAT=true
            shift ;;  # Remove --skip-format from arguments
        --skip-training)
            SKIP_TRAINING=true
            shift ;;
        --skip-scoring)
            SKIP_SCORING=true
            shift ;;
        --skip-combine-sort)
            SKIP_COMBINE_SORT=true
            shift ;;
        --skip-select-topk)
            SKIP_SELECT_TOPK=true
            shift ;;
        *)
            echo "Unknown option: $arg"
            exit 1
            ;;
    esac
done

echo "Starting the pipeline..."

# Step 1: Format
if [ "$SKIP_FORMAT" = false ]; then
    echo "Step 1: Formatting data..."
    python scripts/format.py configs/dclm/format.yaml
    if [ $? -ne 0 ]; then
        echo "Error in formatting step. Exiting."
        exit 1
    fi
else
    echo "Skipping formatting step."
fi

export CHECKPOINTS_PATH=/n/netscratch/sham_lab/Everyone/dclm/color_filter/dclm-pipeline-checkpointing-test

# Step 2: Training
if [ "$SKIP_TRAINING" = false ]; then
    echo "Step 2: Training models..."
    sbatch scripts/launch_sweep.sh configs/sweeps/pretrain.yaml 
    sbatch scripts/launch_sweep.sh configs/sweeps/finetune-dclm.yaml 
else
    echo "Skipping training step."
fi

# Step 3: Scoring
if [ "$SKIP_SCORING" = false ]; then
    echo "Step 3: Scoring..."
    sbatch scripts/launch_sweep.sh configs/sweeps/score-dclm.yaml
else
    echo "Skipping scoring step."
fi

# Step 4: Combine-Sort
if [ "$SKIP_COMBINE_SORT" = false ]; then
    echo "Step 4: Combining and sorting..."
    python scripts/combine_sort.py configs/dclm/combine_sort.yaml
    if [ $? -ne 0 ]; then
        echo "Error in combine-sort step. Exiting."
        exit 1
    fi
else
    echo "Skipping combine-sort step."
fi

# Step 5: Select-TopK
if [ "$SKIP_SELECT_TOPK" = false ]; then
    echo "Step 5: Selecting top k (or documents)..."
    python scripts/select_topk.py configs/dclm/select_topk.yaml
    if [ $? -ne 0 ]; then
        echo "Error in select-topk step. Exiting."
        exit 1
    fi
else
    echo "Skipping select-topk step."
fi

echo "Pipeline completed successfully."
