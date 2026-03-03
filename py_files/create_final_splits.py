from dotenv import load_dotenv
import os

load_dotenv()
processed_path = os.getenv("DATA_PROCESSED_DIR")

# Load filtered data
with open(f"{processed_path}/EMEA.en-fr.en-filtered.en.semantic.en") as f: en_lines = f.readlines()
with open(f"{processed_path}/EMEA.en-fr.fr-filtered.fr.semantic.fr") as f: fr_lines = f.readlines()

print(f"Total pairs after semantic filtering: {len(en_lines):,}")

# Split into 20k train, 1k val
train_en = en_lines[:20000]
train_fr = fr_lines[:20000]
val_en = en_lines[20000:21000]
val_fr = fr_lines[20000:21000]

# Save splits
with open(f"{processed_path}/train/train.en", "w") as f: f.writelines(train_en)
with open(f"{processed_path}/train/train.fr", "w") as f: f.writelines(train_fr)
with open(f"{processed_path}/val/val.en", "w") as f: f.writelines(val_en)
with open(f"{processed_path}/val/val.fr", "w") as f: f.writelines(val_fr)