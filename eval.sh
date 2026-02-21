#!/usr/bin/bash

# --- Basic Configuration ---
GPU=0
MODEL="convnext_t3"       # Model architecture: convnext_t2, t3, t4
CKPT="results/runs-convnext-432/fold1/best.pth" # Path to model weight file
DATA="data/test-data244"     # Path to test dataset
INDICES="4 3 2"           # Image indices corresponding to the model input

# --- Execute Inference ---
python eval.py \
    --test_dir "$DATA" \
    --ckpt "$CKPT" \
    --model_type "$MODEL" \
    --image_names $INDICES \
    --gpu $GPU \
    --out_csv "results.csv"

echo "Done! Results saved to results.csv"