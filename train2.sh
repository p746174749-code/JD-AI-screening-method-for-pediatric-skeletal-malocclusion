#!/usr/bin/bash

# --- 1. Project Configuration ---
# Example: Multi-input experiment using 3 images (indices 4, 3, 2)
MODEL_TYPE="ConvNeXt_3Input"
RUN_NAME="runs/${MODEL_TYPE}_Experiment"
# Generate timestamp once outside the loop to keep all 5 folds in the same directory
TS=$(date +"%Y-%m-%d_%H-%M")

# --- 2. Basic Training Parameters ---
NUM_CLASSES=3
EPOCHS=100
BATCH_SIZE=32
GPU=0
LR=0.0001
# List of image indices to be loaded per sample (e.g., "4 3 2" for 4.jpg, 3.jpg, 2.jpg)
IMAGE_INDICES="4 3 2"

# --- 3. Learning Rate and Early Stopping Hyperparameters ---
WARMUP=20
MIN_LR_RATIO=0.01
PATIENCE=20
STOP_START_EPOCH=40

# --- 4. Print Training Parameters for Verification ---
echo "================================================================"
echo "      MULTI-INPUT TRAINING PARAMETERS INITIALIZED"
echo "================================================================"
echo "Run Name          : $RUN_NAME"
echo "Timestamp         : $TS"
echo "Target Model      : $MODEL_TYPE"
echo "Number of Classes : $NUM_CLASSES"
echo "Total Epochs      : $EPOCHS"
echo "Batch Size        : $BATCH_SIZE"
echo "GPU Device ID     : $GPU"
echo "Initial LR        : $LR"
echo "Image Indices     : $IMAGE_INDICES"
echo "Warmup Epochs     : $WARMUP"
echo "Min LR Ratio      : $MIN_LR_RATIO"
echo "Patience          : $PATIENCE"
echo "Early Stop Start  : $STOP_START_EPOCH"
echo "================================================================"

# Loop through 5-fold cross-validation
for FOLD in {1..5}
do
    DATASET_DIR="data/data-1/fold_${FOLD}"
    echo ">>>> Starting Training: Fold $FOLD ..."

    # Execute Python training script for multi-input
    # '|| exit 1' ensures the loop terminates if a Python error is encountered
    python train2.py --dataset_dir "$DATASET_DIR" \
                    --num_classes $NUM_CLASSES \
                    --epochs $EPOCHS \
                    --batch_size $BATCH_SIZE \
                    --learning_rate $LR \
                    --gpu $GPU \
                    --image_indices $IMAGE_INDICES \
                    --run_name "$RUN_NAME" \
                    --timestamp "$TS" \
                    --fold "fold_${FOLD}" \
                    --warmup_epochs $WARMUP \
                    --min_lr_ratio $MIN_LR_RATIO \
                    --use_early_stop \
                    --early_stop_start $STOP_START_EPOCH \
                    --patience $PATIENCE || exit 1

    echo ">>>> Fold $FOLD Finished Successfully."
done

echo "----------------------------------------------------------------"
echo "Experiment Process Completed."
echo "Results stored in: $RUN_NAME/$TS/"
echo "----------------------------------------------------------------"