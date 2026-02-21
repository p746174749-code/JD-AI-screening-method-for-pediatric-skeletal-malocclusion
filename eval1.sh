#!/usr/bin/bash

# --- Configuration ---
GPU=0
MODEL_ROOT="results/runs-convnext-432"  # Root folder containing fold1, fold2, etc.
TEST_DATA="data/test-data244"
IMG_INDICES="4 3 2"
OUT_FOLDER="results_avg"

# --- Execute Inference ---
# The Python script will automatically find fold*/best.pth under MODEL_ROOT
python eval1.py \
    --test_dir "$TEST_DATA" \
    --ckpt_paths "$MODEL_ROOT" \
    --image_names $IMG_INDICES \
    --out_dir "$OUT_FOLDER" \
    --num_classes 3 \
    --gpu_id $GPU \
    --batch_size 1

echo "Ensemble process finished."