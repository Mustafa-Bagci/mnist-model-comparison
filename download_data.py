import kagglehub
import shutil
import os

# Define the target directory
target_dir = "MNIST Datasets"

# Download the MNIST dataset
source_path = kagglehub.dataset_download("hojjatk/mnist-dataset")

# Ensure the target directory exists
os.makedirs(target_dir, exist_ok=True)

# Move the downloaded files to the target directory
for file in os.listdir(source_path):
    shutil.move(os.path.join(source_path, file), target_dir)

print(f"Dataset downloaded and moved to: {target_dir}")
