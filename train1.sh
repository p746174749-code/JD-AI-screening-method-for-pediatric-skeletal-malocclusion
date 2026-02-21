#!/usr/bin/bash

# --- 1. Model Architecture Configuration ---
# Options: resnet152, resnet101, densenet121, convnext_t, convnext_s,  convnext_b, swin_t, swin_s, swin_b, vit
MODEL_TYPE="convnext_t"

# --- 2. Project Path Configuration ---
RUN_NAME="runs/${MODEL_TYPE}_SingleInput_Experiment"
# Generate timestamp once outside the loop to keep all folds in the same directory
TS=$(date +"%Y-%m-%d_%H-%M")

# --- 3. Basic Training Parameters ---
NUM_CLASSES=3
EPOCHS=100
BATCH_SIZE=32
GPU=1
LR=0.0001
# Single input usually selects one index, e.g., "4" for 4.jpg, "3" for 3.jpg
IMAGE_INDICES="4"

# --- 4. Optimizer & Early Stopping Hyperparameters ---
WARMUP=20
MIN_LR_RATIO=0.01
PATIENCE=20
STOP_START_EPOCH=40

# --- 5. Print Training Parameters for Verification ---
echo "================================================================"
echo "     SINGLE-INPUT TRAINING PARAMETERS INITIALIZED"
echo "================================================================"
echo "Model Type      : $MODEL_TYPE"
echo "Run Name        : $RUN_NAME"
echo "Timestamp       : $TS"
echo "Classes         : $NUM_CLASSES"
echo "Epochs          : $EPOCHS"
echo "Batch Size      : $BATCH_SIZE"
echo "GPU ID          : $GPU"
echo "Learning Rate   : $LR"
echo "Image Indices   : $IMAGE_INDICES"
echo "Warmup Epochs   : $WARMUP"
echo "Min LR Ratio    : $MIN_LR_RATIO"
echo "Patience        : $PATIENCE"
echo "Early Stop Start: $STOP_START_EPOCH"
echo "================================================================"

# Loop through 5-fold cross-validation
for FOLD in {1..5}
do
    DATASET_DIR="data/data-1/fold_${FOLD}"
    echo ">>>> Starting Training: Fold $FOLD using $MODEL_TYPE ..."

    # Execute Python training script
    # Added '|| exit 1' to stop the script if a Python error occurs
    python train1.py --dataset_dir "$DATASET_DIR" \
                    --model_type "$MODEL_TYPE" \
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
echo "All Folds Completed."
echo "Final Results Directory: $RUN_NAME/$TS/"
echo "----------------------------------------------------------------"