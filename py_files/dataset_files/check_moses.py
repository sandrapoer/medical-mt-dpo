from dotenv import load_dotenv
import os

load_dotenv()
raw_path = os.getenv("DATA_RAW_DIR")

# Count lines in each file
with open(f"{raw_path}/EMEA.en-fr.en") as f:
    emea_count = sum(1 for _ in f)

with open(f"{raw_path}/ELRC-2720-EMEA.en-fr.en") as f:
    elrc_count = sum(1 for _ in f)

print(f"EMEA v3: {emea_count:,} sentences")
print(f"ELRC-2720-EMEA v1: {elrc_count:,} sentences")

