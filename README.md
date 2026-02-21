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
To ensure the multi-view fusion mechanism works correctly, the dataset must follow a specific directory structure.
1. Dataset Directory Structure
The model expects a nested folder structure where each sample (case) contains its respective view images.
```text
data/
├── train/                # Training set
│   ├── class_0/          # Category name
│   │   ├── 001/          # Unique ID for each case
│   │   │   ├── 4.jpg     # 90° profile view
│   │   │   ├── 3.jpg     # 45° profile view
│   │   │   └── 2.jpg     # frontal smile view
│   │   │   └── 1.jpg     # frontal view
│   │   └── 002/
│   └── class_1/
│   ├── class_2/
└── val/                  # Training set       
```

