"""Convert TICO-19 TSV to messages jsonl (plain + terms).

Creates:
  data/processed/test/messages_test.jsonl        (plain prompts)

The terms version is created by adding test/ to term_instructions.py SPLITS
and rerunning it — same pattern as test_emea. See printed instructions at end.

Run once before final evaluation.
"""
import os
import json
import pandas as pd
from dotenv import load_dotenv

load_dotenv()
RAW  = os.getenv("DATA_RAW_DIR").rstrip("/")
PROC = os.getenv("DATA_PROCESSED_DIR").rstrip("/")

PLAIN_INSTRUCTION = (
    "Translate the English source text to French. "
    "Rules: output strictly the French translation. "
    "Do NOT repeat the English text. "
    'Do NOT include labels like "English" or "French". '
    "Do NOT add quotation marks, explanations or any extra text. "
    "Now translate the following:\nEnglish: {source}\nFrench:\n"
)

tsv  = pd.read_csv(f"{RAW}/test.en-fr.tsv", sep="\t")
src  = tsv["sourceString"].tolist()
tgt  = tsv["targetString"].tolist()
print(f"TICO-19 pairs: {len(src)}")

out = f"{PROC}/test"
os.makedirs(out, exist_ok=True)

with open(f"{out}/messages_test.jsonl", "w", encoding="utf-8") as f:
    for s, t in zip(src, tgt):
        rec = {"messages": [
            {"role": "user",      "content": PLAIN_INSTRUCTION.format(source=str(s).strip())},
            {"role": "assistant", "content": str(t).strip()},
        ]}
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")

# Also write raw .en/.fr if not already present (used by sacrebleu CLI).
en_path = f"{out}/test.en"
fr_path = f"{out}/test.fr"
if not os.path.isfile(en_path):
    with open(en_path, "w") as f:
        f.write("\n".join(str(s).strip() for s in src) + "\n")
if not os.path.isfile(fr_path):
    with open(fr_path, "w") as f:
        f.write("\n".join(str(t).strip() for t in tgt) + "\n")

print(f"Wrote: {out}/messages_test.jsonl")
print("\nFor the TERMS version add this to SPLITS in term_instructions.py and rerun:")
print(f'''
    {{
        "en":     DATA_PROCESSED_DIR / "test/test.en",
        "msgs":   DATA_PROCESSED_DIR / "test/messages_test.jsonl",
        "output": DATA_PROCESSED_DIR / "test/messages_test_terms.jsonl",
    }},
''')