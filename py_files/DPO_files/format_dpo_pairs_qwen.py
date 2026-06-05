import json, os, re
from pathlib import Path
from dotenv import load_dotenv
from transformers import AutoTokenizer

load_dotenv()
DATA_PROCESSED_DIR = Path(os.getenv("DATA_PROCESSED_DIR"))

IN_PATH  = DATA_PROCESSED_DIR / "dpo/umls_dpo_pairs_qwen.jsonl"
OUT_PATH = DATA_PROCESSED_DIR / "dpo/dpo_train_qwen.jsonl"

INSTRUCTION = "Translate the following text from English into French.\n"

def strip_think(text: str) -> str:
    return re.sub(r"<think>.*?</think>\n?", "", text, flags=re.DOTALL).strip()

written = 0
with open(IN_PATH) as fin, open(OUT_PATH, "w") as fout:
    for line in fin:
        rec = json.loads(line)
        source  = rec["source"]
        chosen  = strip_think(rec["chosen"])
        rejected = strip_think(rec["rejected"])

        prompt = (
            f"<|im_start|>user\n{INSTRUCTION}{source}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )

        fout.write(json.dumps({
            "prompt":   prompt,
            "chosen":   chosen,
            "rejected": rejected,
        }, ensure_ascii=False) + "\n")
        written += 1

print(f"Written: {written} pairs to {OUT_PATH}")
