# This repository contains the official implementation of the study: **"AI screening method for pediatric skeletal malocclusion using facial characteristics"**.
PSM-MVFF: Pediatric Skeletal Malocclusion via Multi-View Feature Fusion

## 🛠️ Installation 
We recommend using Miniconda or Anaconda to manage your Python environment to avoid dependency conflicts.
This project requires `Python 3.9` and `CUDA 11.6` We recommend using conda to manage dependencies.

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name
```

### 2. Create Environment
```bash
conda create -n multiview python=3.9
conda activate multiview
```

### 3. Install PyTorch (CUDA 11.6)
Since the project uses a specific `CUDA` version, please install `PyTorch` using the following command:
```bash
pip install torch==1.13.0+cu116 torchvision==0.14.0+cu116 torchaudio==0.13.0+cu116 --extra-index-url https://download.pytorch.org/whl/cu116
```

### 4. Install Other Dependencies
Install the remaining libraries from the `requirements.txt`:
```bash
pip install -r requirements.txt
```

## 📂 Date Preparation
The data preparation process consists of two stages: organizing the raw images and performing the 5-Fold Cross-Validation split.

### 1. Raw Data Organization
Before splitting, ensure your original dataset is stored in data/data-raw. Each sample (Case ID) folder must contain four specific view images.
```text
data/data-raw/
├── class_0/                 # Category 0 folder
│   ├── 001/                 # Unique Case ID
│   │   ├── 1.jpg            # Frontal view
│   │   ├── 2.jpg            # Frontal smile view
│   │   ├── 3.jpg            # 45° profile view
│   │   └── 4.jpg            # 90° profile view
│   ├── 002/
│   └── ...
├── class_1/
└── class_2/
```
### 2. 5-Fold Cross-Validation Split
We use `utlis/dataset_split.py` to randomly partition the raw data into 5 folds while maintaining the class distribution.
Run the splitting script:
```bash
python utlis/dataset_split.py
```
Output Structure (data/data-fold/):
The script generates 5 folders (fold_1 to fold_5). Each fold contains independent train and val sets:
```text
data/data-fold/
├── fold_1/                  # Fold 1 experimental data
│   ├── train/               # Training Set (80% of data)
│   │   ├── class_0/
│   │   │   ├── 001/
│   │   │   │   ├── 1.jpg    # frontal view
│   │   │   │   ├── 2.jpg    # frontal smile view
│   │   │   │   ├── 3.jpg    # 45° profile view
│   │   │   │   ├── 4.jpg    # 90° profile view
│   │   │   └── ...
│   │   └── class_1
│   │   └── class_2
│   └── val/                 # Validation Set (20% of data)
│       └── ...
├── fold_2/                  # Fold 2 experimental data
├── ...                      # Fold 3, Fold 4
└── fold_5/                  # Fold 5 experimental data      
```

## 🏋️ Training Guide

Our framework provides a highly configurable **Single-View Pipeline** and a **Multi-View Fusion Pipeline**. Both support automated **5-Fold Cross-Validation** to ensure robust and reproducible experimental results.

---

### 1. Single-View Models Training (`train1.py`)

This pipeline allows for training on any specific view (e.g., just the 90° profile or just the frontal view) using various state-of-the-art architectures.

* **Multiple Architectures**: Native support for `ResNet`, `DenseNet`, `ConvNeXt (T/S/B)`, `Swin Transformer`, and `Vision Transformer (ViT)`.
* **Customizable Input**: Change the `IMAGE_INDICES` in the shell script to select which specific `.jpg` to use as the input.
* **Advanced Scheduling**: Features Learning Rate Warmup, Cosine Annealing, and Early Stopping logic.


**To run the 5-fold training:**

```bash
# 1. Configure MODEL_TYPE and IMAGE_INDICES inside the script first
chmod +x scripts/train1.sh

# 2. Execute the training
./scripts/train1.sh
```

---

### 2. Multi-View Models Training (`train2.py`)

This pipeline implements our proposed **Primary-Auxiliary Feature Fusion Strategy**. Unlike standard multi-input models, our architecture distinguishes between the primary view and auxiliary views to enhance feature representation.

* **Primary Feature**: View `4.jpg` (90° profile) is treated as the anchor/primary input.
* **Auxiliary Features**: Views `3.jpg`, `2.jpg`, or `1.jpg` are processed as auxiliary inputs to provide complementary spatial information.
* **Modular Design**: By modifying the `IMAGE_INDICES` in the script, you can dynamically adjust the number of inputs (e.g., 2-view, 3-view, or 4-view fusion).


**To run the 5-fold training:**

```bash
# 1. Configure MODEL_TYPE and IMAGE_INDICES inside the script first
chmod +x scripts/train2.sh

# 2. Execute the training
./scripts/train2.sh
```

---

### 📈 Monitoring & Logging


All training logs, model checkpoints, and hyperparameter configs are saved in the `runs/` directory, organized by **Model Type** and **Timestamp**. 

* **TensorBoard**: Visualize loss and accuracy curves in real-time.
    ```bash
    tensorboard --logdir=runs/
    ```
* **Checkpoints**: The best model for each fold is automatically saved as `best.pth` within its respective fold directory.


## 🔍 Evaluation & Results

Before running the evaluation scripts, ensure your test data is organized in the same directory structure as the training data. The script `eval1.py` and `eval2.py` will iterate through each numbered folder to perform batch inference.

**Directory Structure (`data/data-test/`):**
Each subfolder represents a single sample (e.g., a patient or a case) and must contain the required image views.

```text
data/data-test/
├── 001/                 # Sample ID
│   ├── 1.jpg            # Frontal view
│   ├── 2.jpg            # Frontal smile view
│   ├── 3.jpg            # 45° profile view
│   └── 4.jpg            # 90° profile view (Primary)
├── 002/
├── 003/
└── ...
```
To evaluate the model performance or perform inference, you first need to prepare the trained weights and run the evaluation scripts.

### 1. Model Weights Preparation
Since the model weights are large, they are stored externally. We provide pre-trained weights for all 5 folds.

1. **Download**: Download the weights from [https://drive.google.com/drive/folders/1dANClACF7XFrnCw92VwSsEW4ZTb-LgHG?usp=drive_link].
2. **Setup**: Create a `checkpoints` directory in the project root and extract the weights into it.
3. **Structure**: Ensure the directory follows this structure for the scripts to correctly locate the models:

```text
checkpoints/
└── runs-convnext-432/   # Example experiment name
    └── fold_1/
        └── best.pth
    └── fold_2/
        └── best.pth
    └── ... (up to fold_5)
```

---

### 2. Single Checkpoint Inference (`eval1.py`)
Use this script to verify the performance of a specific model weight file. It supports both single-view and multi-view architectures.

**How to run:**
```bash
python eval1.py --test_dir data/data-test \
                --ckpt checkpoints/runs-convnext-432/fold1/best.pth \
                --model_type convnext_t3 \
                --image_names 4 3 2 \
                --gpu 0 \
                --out_csv results_single.csv
```

### 3. 5-Fold Ensemble Evaluation (`eval2.py`)
For maximum robustness, `eval2.py` implements a **Soft Voting Strategy**. It aggregates the class probabilities from all five folds to produce a final, fused prediction.

1. **Auto-Discovery**: Automatically finds all `best.pth` files within a specified directory.
2. **Probability Averaging**: Reduces model bias by calculating the mean probability across all folds.
3. **Detailed Export**: Saves individual results for each model and a final ensemble file.

**How to run:**
```bash
python eval2.py --test_dir data/data-test \
                --ckpt_paths checkpoints/runs-convnext-432 \
                --image_names 4 3 2 \
                --out_dir results_ensemble_5fold \
                --gpu_id 0
```

When you run the 5-fold ensemble script, the `--out_dir` will be populated with several CSV files. This allows you to analyze both the consistency of individual folds and the final fused performance.

**Output Directory Structure:**
```text
results/
├── model1.csv           # Predictions & probabilities from Fold 1
├── model2.csv           # Predictions & probabilities from Fold 2
├── ...
├── model5.csv           # Predictions & probabilities from Fold 5
└── avg.csv     # Final fused results (The "Main" Result)
```

#### CSV Column Definitions:
Each file (both individual models and the ensemble) contains the following columns:

| Column Name | Description |
| :--- | :--- |
| `id` | The sample ID (derived from the subfolder name). |
| `pred` | The final predicted class label (mapped to 1, 2, or 3). |
| `prob_class1` | Confidence score (Softmax probability) for Class 1. |
| `prob_class2` | Confidence score (Softmax probability) for Class 2. |
| `prob_class3` | Confidence score (Softmax probability) for Class 3. |

---
---

### 🚀 Quick Start Script
You can also use the provided shell script to run the inference pipeline with one command:
```bash
chmod +x scripts/eval1.sh
./scripts/eval1.sh

chmod +x scripts/eval2.sh
./scripts/eval2.sh
```


## 📊 Metrics Calculation & Statistical Analysis

Once the ensemble inference is complete, we use the specific metrics script to perform a professional statistical analysis of the 5-fold cross-validation results.

### 1. Data Preparation
To run the analysis, you need to consolidate the Ground Truth and all fold predictions into a single Excel file (e.g., `data-test-pred-results.xlsx`).

**Required File Structure:**
| id | GT | F1 | F2 | F3 | F4 | F5 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 101 | 1 | 1 | 1 | 2 | 1 | 1 |
| 102 | 3 | 3 | 3 | 3 | 3 | 3 |

---

### 2. Statistical Analysis Script (`calculate_metrics.py`)

This script goes beyond simple accuracy. It calculates per-class metrics (recall, Specificity, F1, Acc) and provides a final report in the **Mean ± Standard Deviation** format, which is standard for academic publications.

* **Recall (Sensitivity)**: Per-class and Macro average.
* **Specificity**: Per-class and Macro average.
* **F1-Score**: Per-class, Macro, and Weighted averages.
* **Overall Accuracy**: The total percentage of correct predictions.

**How to run:**
```bash
# Ensure you have pandas, openpyxl, and scikit-learn installed
python calculate_metrics.py
```

**Example Terminal Output:**
```text
==================================================
     Performance Metric Summary (Mean ± Std)      
==================================================
Recall (C1)          : 68.30 ± 4.08
Recall (C2)          : 83.25 ± 4.81
Recall (C3)          : 84.86 ± 5.01
Macro recall         : 78.80 ± 1.34
Specificity (C1)     : 85.87 ± 3.69
Specificity (C2)     : 87.32 ± 2.42
Specificity (C3)     : 93.22 ± 1.25
Macro Specificity    : 88.80 ± 0.67
F1 (C1)              : 71.55 ± 1.31
F1 (C2)              : 79.55 ± 2.47
F1 (C3)              : 84.09 ± 1.87
Macro F1             : 78.40 ± 1.17
Weighted F1          : 77.77 ± 1.20
Overall Accuracy     : 77.95 ± 1.24
==================================================
```

* **`Detailed_Performance_Table.xlsx`**: Contains all raw metrics for each of the 5 folds plus the final Mean ± Std column. Perfect for deep-diving into which fold performed best.
* 
## 🎨 Advanced Visualization (ROC & PR Curves)

To evaluate the discriminative power of the 5-fold ensemble model, we provide a high-quality visualization script `plot_curves.py`. This script generates **Receiver Operating Characteristic (ROC)** and **Precision-Recall (PR)** curves based on the averaged probabilities.

### 1. Data Merging Requirement
The visualization script requires a consolidated CSV file (e.g., `avg.csv`). You must take the `avg_ensemble.csv` generated by `eval2.py` and ensure it includes a `GT` (Ground Truth) column.

**Target File Format (`avg.csv`):**
| id | GT | pred | prob_class1 | prob_class2 | prob_class3 |
| :---| :--- | :--- | :--- | :--- | :--- |
| 001 | 1 | 1 | 0.95 | 0.03 | 0.02 |
| 002 | 3 | 3 | 0.01 | 0.10 | 0.89 |

---

### 2. Generating High-Quality Figures

The script uses `matplotlib` with professional styling suitable for academic papers. It calculates:
* **Micro-average**: Aggregates the contributions of all classes to compute the average metric.
* **Macro-average**: Computes the metric independently for each class and then takes the average (treating all classes equally).

**How to run:**
```bash
# Ensure the INPUT_FILE name in the script matches your merged CSV
python plot_curves.py
```

---

## 🧠 Explainability & Visual Interpretability (Grad-CAM)

To validate that the model makes decisions based on clinically relevant facial features (and not background noise), we implement **Grad-CAM**. This technique generates heatmaps highlighting the regions that most strongly influenced the AI's classification.



### 1. Visualizing Model Attention (`run_gradcam.py`)

The script uses a hook-based approach to extract feature maps and gradients from the last convolutional layer of the **ConvNeXt-T** backbone. It then overlays a Jet-color heatmap onto the original facial photographs.

* **Multi-View Support**: Generates heatmaps for all input views (90° Profile, 45° Profile, and Frontal Smile).
* **Automated Batch Processing**: Iterates through the test directory and saves results for each sample.

### 2. How to Run Visualization

```bash
# Update the MODEL_WEIGHTS and DATA_DIR paths in the script first
python run_gradcam.py
```

---

### 🖼️ Interpreting Grad-CAM Results

The output will be saved in the `results_gradcam/` directory, organized by sample ID. 

**Example output for one sample:**
* `gradcam_x1.png` (90° Profile View)
* `gradcam_x2.png` (45° Profile View)
* `gradcam_x3.png` (Frontal Smile View)


---

