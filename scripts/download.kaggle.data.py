from pathlib import Path
import shutil
import pandas as pd
import kagglehub

# Detect project root
PROJECT_ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()

RAW_DIR = PROJECT_ROOT / "data" / "raw" / "nih_chest_xray"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# Download dataset
download_path = Path(kagglehub.dataset_download("nih-chest-xrays/data"))
print("Downloaded to:", download_path)

# Copy dataset into project, only if it is not already there
if not RAW_DIR.exists():
    print("Copying dataset into data/raw/nih_chest_xray. This may take a while...")
    shutil.copytree(download_path, RAW_DIR)
else:
    print("Raw dataset already exists:", RAW_DIR)