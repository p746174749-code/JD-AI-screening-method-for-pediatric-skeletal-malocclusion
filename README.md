The paper is currently under review, and the code will be released subsequently.

🛠️ Installation 
We recommend using Miniconda or Anaconda to manage your Python environment to avoid dependency conflicts.
This project requires Python 3.8+ and CUDA 11.6. We recommend using conda to manage dependencies.

1. Clone the Repository
'''
Bash
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name

3. Create Environment
Bash
conda create -n multiview python=3.8 -y
conda activate multiview

5. Install PyTorch (CUDA 11.6)
Since the project uses a specific CUDA version, please install PyTorch using the following command:
Bash
pip install torch==1.13.0+cu116 torchvision==0.14.0+cu116 torchaudio==0.13.0+cu116 --extra-index-url https://download.pytorch.org/whl/cu116

6. Install Other Dependencies
Install the remaining libraries (timm, opencv, pandas, etc.) from the requirements.txt:
Bash
pip install -r requirements.txt
