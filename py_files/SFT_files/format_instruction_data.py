from dotenv import load_dotenv
import os
import json

load_dotenv()
processed_path = os.getenv("DATA_PROCESSED_DIR")

# Instruction Template
INSTRUCTION_TEMPLATE = """
Translate the {src} source text to {tgt}. Rules: output strictly the {tgt} translation. Do NOT repeat the {src} text. Do NOT include labels like "{src}" or "{tgt}". Do NOT add quotation marks, explanations or any extra text. Now translate the following:
{src}: {source}
{tgt}:
"""

def format_pair(en, fr):
    prompt = INSTRUCTION_TEMPLATE.format(
        src="English",
        tgt="French",
        source=en.strip(),
    )
    return {"prompt": prompt, "completion": fr.strip()}

# Load Data
with open(f"{processed_path}/val/val.en") as f:
    train_en = f.readlines()
with open(f"{processed_path}/val/val.fr") as f:
    train_fr = f.readlines()

# Format Data
formatted_data = [format_pair(en, fr) for en, fr in zip(train_en, train_fr)]

with open(f"{processed_path}/val/formatted_val.jsonl", "w") as f:
    for item in formatted_data:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")

print(f"Formatted {len(formatted_data)} example pairs.")
print(f"Formatted data saved to {processed_path}/val/formatted_val.json")
print(formatted_data[0])