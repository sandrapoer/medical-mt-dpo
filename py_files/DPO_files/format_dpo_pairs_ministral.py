import json, os, re
from pathlib import Path
from dotenv import load_dotenv
from transformers import AutoTokenizer

load_dotenv()
DATA_PROCESSED_DIR = Path(os.getenv("DATA_PROCESSED_DIR"))

IN_PATH  = DATA_PROCESSED_DIR / "dpo/umls_dpo_pairs_ministral.jsonl"
OUT_PATH = DATA_PROCESSED_DIR / "dpo/dpo_train_ministral.jsonl"

INSTRUCTION = "Translate the following text from English into French.\n"

tok = AutoTokenizer.from_pretrained("mistralai/Ministral-8B-Instruct-2410", trust_remote_code=True)

written = 0
with open(IN_PATH) as fin, open(OUT_PATH, "w") as fout:
    for line in fin:
        rec = json.loads(line)
        source   = rec["source"]
        chosen   = rec["chosen"]
        rejected = rec["rejected"]

        prompt = tok.apply_chat_template(
            [{"role": "user", "content": f"{INSTRUCTION}{source}"}],
            tokenize=False,
            add_generation_prompt=True,
        )

        fout.write(json.dumps({
            "prompt":   prompt,
            "chosen":   chosen,
            "rejected": rejected,
        }, ensure_ascii=False) + "\n")
        written += 1

print(f"Written: {written} pairs to {OUT_PATH}")
