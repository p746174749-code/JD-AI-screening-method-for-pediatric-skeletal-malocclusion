The paper is currently under review, and the code will be released subsequently.

🛠️ Installation 
We recommend using Miniconda or Anaconda to manage your Python environment to avoid dependency conflicts.
This project requires Python 3.8+ and CUDA 11.6. We recommend using conda to manage dependencies.

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

📂 Data Preparation
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

