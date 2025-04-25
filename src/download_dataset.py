import os
import kaggle

root_dir = os.getcwd()

if root_dir.endswith("scripts"):
    # If the current working directory ends with "scripts", go up one level
    root_dir = os.path.dirname(root_dir)

data_dir = os.path.join(root_dir, "data")
dataset_name = "lexset/synthetic-asl-alphabet"
unzip = True

# Make sure to have your Kaggle API key saved in your home directory
# as ~/.kaggle/kaggle.json or C:\Users\<username>\.kaggle\kaggle.json
kaggle.api.authenticate()

os.makedirs(data_dir, exist_ok=True)

print(f"Downloading dataset: `{dataset_name}` ...") 
kaggle.api.dataset_download_files(dataset_name, path=data_dir, unzip=unzip)

print("✅ Dataset downloaded successfully.")
