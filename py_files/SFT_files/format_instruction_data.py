from dotenv import load_dotenv
import os
import json

load_dotenv()
processed_path = os.getenv("DATA_PROCESSED_DIR").rstrip("/")

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
    return {
        "messages": [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": fr.strip()}
        ]
    }

def format_split(src_file, tgt_file, out_file):
    with open(src_file, encoding="utf-8") as f:
        en_lines = f.readlines()
    with open(tgt_file, encoding="utf-8") as f:
        fr_lines = f.readlines()

    formatted = [format_pair(en, fr) for en, fr in zip(en_lines, fr_lines)]

    with open(out_file, "w", encoding="utf-8") as f:
        for item in formatted:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"Formatted {len(formatted):,} pairs -> {out_file}")
    print("Example:", formatted[0])
    print()

# Train
format_split(
    src_file=f"{processed_path}/train/train.en",
    tgt_file=f"{processed_path}/train/train.fr",
    out_file=f"{processed_path}/train/messages_train.jsonl",
)

# Val
format_split(
    src_file=f"{processed_path}/val/val.en",
    tgt_file=f"{processed_path}/val/val.fr",
    out_file=f"{processed_path}/val/messages_val.jsonl",
)