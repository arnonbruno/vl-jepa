"""
Download Conceptual Captions 12M metadata and prepare dataset for training.
"""

import os
import subprocess
import json
from pathlib import Path

print("=" * 70)
print("Downloading Conceptual Captions 12M")
print("=" * 70)

# Create data directory
data_dir = Path.home() / 'cc12m'
data_dir.mkdir(exist_ok=True)

# Download the 2.12GB metadata file
metadata_url = "https://huggingface.co/datasets/laion/conceptual-captions-12m-webdataset/resolve/main/cc12m-train-000000.tar"

print(f"\n📥 Downloading CC12M metadata (~2GB)...")
print(f"Destination: {data_dir}/cc12m_metadata.tar")

# Use wget for large file download
cmd = f"cd {data_dir} && wget -c {metadata_url} -O cc12m_train.tar"

result = subprocess.run(cmd, shell=True, capture_output=False)

if result.returncode == 0:
    print(f"\n✓ Downloaded successfully")
    print(f"Extracting...")
    
    # Extract the tar file
    extract_cmd = f"cd {data_dir} && tar -xf cc12m_train.tar"
    subprocess.run(extract_cmd, shell=True)
    
    print(f"✓ Extracted to {data_dir}")
    print(f"\nNext: Run exp_07_cc12m.py to train on CC12M data")
else:
    print(f"\n✗ Download failed. Trying alternative mirror...")
    
    # Try GitHub mirror
    alt_url = "https://raw.githubusercontent.com/google-research-datasets/conceptual-12m/master/train_urls.txt"
    alt_cmd = f"cd {data_dir} && wget {alt_url} -O cc12m_urls.txt"
    subprocess.run(alt_cmd, shell=True)
    
    print(f"Downloaded URL list to {data_dir}/cc12m_urls.txt")
    print(f"You can now manually download images or use a parallel downloader")
