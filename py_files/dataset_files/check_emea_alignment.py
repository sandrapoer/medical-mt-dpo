from dotenv import load_dotenv
import os

load_dotenv()
raw_path = os.getenv("DATA_RAW_DIR")

# Load first 20 pairs
with open(f"{raw_path}/EMEA.en-fr.en") as f_en:
    en_lines = [next(f_en) for _ in range(20)]

with open(f"{raw_path}/EMEA.en-fr.fr") as f_fr:
    fr_lines = [next(f_fr) for _ in range(20)]

for i, (en, fr) in enumerate(zip(en_lines, fr_lines), 1):
    print(f"{i}.")
    print(f"EN: {en.strip()}")
    print(f"FR: {fr.strip()}")