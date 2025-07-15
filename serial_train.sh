#!/bin/bash
set -e

# Path to datasets
DATASET_DIR="/workspace/nerf_synthetic"

# Base command components
SCRIPT_PATH="/workspace/permuto_sdf/permuto_sdf_py/experiments/run_custom_dataset/run_custom_test.py"
BASE_CMD="python3 -u $SCRIPT_PATH --dataset custom --scene_scale 0.25 --no_viewer --with_mask"
EXP_ROOT="mask_test"
LOG_DIR="workspace/checkpoint_copy/logs"
mkdir -p "$LOG_DIR"
# Loop over each folder in dataset
for folder in "$DATASET_DIR"/*; do
    if [ -d "$folder" ]; then
        dataset_name=$(basename "$folder")
        echo "Starting training on: $dataset_name"
        echo "Started at: $(date)"

        EXP_INFO="${EXP_ROOT}_${dataset_name}"

        $BASE_CMD --dataset_path "$folder" --exp_info "$EXP_INFO" | tee "$LOG_DIR/${EXP_INFO}_log.txt"

        echo "Finished training on: $dataset_name"
        echo "-----------------------------------"
    fi
done
