The paper is currently under review, and the code will be released subsequently.


## 🛠️ Installation 
We recommend using Miniconda or Anaconda to manage your Python environment to avoid dependency conflicts.
This project requires `Python 3.8+` and `CUDA 11.6.` We recommend using conda to manage dependencies.

1. Clone the Repository

```bash
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name
```

3. Create Environment
```bash
conda create -n multiview python=3.8 -y
conda activate multiview
```

4. Install PyTorch (CUDA 11.6)
Since the project uses a specific CUDA version, please install PyTorch using the following command:
```bash
pip install torch==1.13.0+cu116 torchvision==0.14.0+cu116 torchaudio==0.13.0+cu116 --extra-index-url https://download.pytorch.org/whl/cu116
```

5. Install Other Dependencies
Install the remaining libraries (timm, opencv, pandas, etc.) from the requirements.txt:
```bash
pip install -r requirements.txt
```

## 📂 Training Guide Date Preparation
The data preparation process consists of two stages: organizing the raw images and performing the 5-Fold Cross-Validation split.
1. Raw Data Organization
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
2. 5-Fold Cross-Validation Split
We use utlis/dataset_split.py to randomly partition the raw data into 5 folds while maintaining the class distribution (stratified-like approach).
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

To evaluate the model performance or perform inference, you first need to prepare the trained weights and run the evaluation scripts.

### 1. Model Weights Preparation
Since the model weights are large, they are stored externally. We provide pre-trained weights for all 5 folds.

1. **Download**: Download the weights from [Your Cloud Drive Link (e.g., Google Drive/Baidu Pan)].
2. **Setup**: Create a `checkpoints` directory in the project root and extract the weights into it.
3. **Structure**: Ensure the directory follows this structure for the scripts to correctly locate the models:

```text
checkpoints/
└── ConvNeXt_3Input_Experiment/   # Example experiment name
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
                --ckpt checkpoints/runs-convnext/fold1/best.pth \
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
                --ckpt_paths checkpoints/ConvNeXt_3Input_Experiment \
                --image_names 4 3 2 \
                --out_dir results-figure \
                --gpu_id 0
```

---


---

### 2. Multi-Fold Ensemble Evaluation
Our evaluation script eval2.py` automatically loads the best weights from all 5 folds, performs ensemble inference (voting/averaging), and generates comprehensive performance reports.

**Key Features:**
* **Metrics**: Calculates Accuracy, Precision, Recall, and F1-score for each fold and the ensemble.
* **Visualization**: Automatically generates **Confusion Matrices** and **ROC Curves**.
* **Report**: Exports all numerical results into a formatted **Excel file** for academic reporting.

**How to run evaluation:**
```bash
python eval/ensemble_eval.py --checkpoint_dir checkpoints/ConvNeXt_3Input_Experiment --data_dir data/data-fold --output_dir results/evaluation_reports
```

---

### 📊 Understanding the Ensemble Logic
The final prediction ($y$) for each sample is determined by averaging the softmax probabilities ($P$) from each fold ($n$):

$$y = \text{argmax} \left( \frac{1}{n} \sum_{i=1}^{n} P_i \right)$$

**Output Files in `results-figure/`:**
* `model1.csv` ... `model5.csv`: Individual fold predictions and probability distributions.
* `avg_ensemble.csv`: The final consensus prediction based on all 5 folds.

---

### 🚀 Quick Start Script
You can also use the provided shell script to run the inference pipeline with one command:
```bash
chmod +x scripts/eval.sh
./scripts/eval.sh
```
