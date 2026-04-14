from dotenv import load_dotenv
import os

load_dotenv()
raw_path = os.getenv("DATA_RAW_DIR")
proc_path = os.getenv("DATA_PROCESSED_DIR")

# Load EMEA 
with open(f"{raw_path}/EMEA.en-fr.en") as f:
    en_lines = f.readlines()
with open(f"{raw_path}/EMEA.en-fr.fr") as f:
    fr_lines = f.readlines()

# Split into 90% train and 10%val
total = len(en_lines)
train_size = int(0.90 * total)

print(f"Total: {total:,}")
print(f"Train: {train_size:,}")
print(f"Val: {total - train_size:,}")

# Create directories
os.makedirs(f"{proc_path}/train", exist_ok=True)
os.makedirs(f"{proc_path}/val", exist_ok=True)

# Save splits
with open(f"{proc_path}/train/train.en", "w") as f:
    f.writelines(en_lines[:train_size])
with open(f"{proc_path}/train/train.fr", "w") as f:
    f.writelines(fr_lines[:train_size])
    
with open(f"{proc_path}/val/val.en", "w") as f:
    f.writelines(en_lines[train_size:])
with open(f"{proc_path}/val/val.fr", "w") as f:
    f.writelines(fr_lines[train_size:])

print("✓ Done!")
