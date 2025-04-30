import os
import kaggle
from pathlib import Path

root_dir = Path(__file__).resolve().parents[2]
data_dir = root_dir / "data"

dataset_name = "lexset/synthetic-asl-alphabet"  # synthetic data
unzip = True

# Make sure to have your Kaggle API key saved in your home directory
# as ~/.kaggle/kaggle.json or C:\Users\<username>\.kaggle\kaggle.json
kaggle.api.authenticate()

os.makedirs(data_dir, exist_ok=True)

print(f"Downloading dataset: `{dataset_name}` ...") 
kaggle.api.dataset_download_files(dataset_name, path=data_dir, unzip=unzip)

print(f"✅ Dataset {dataset_name} downloaded successfully.")