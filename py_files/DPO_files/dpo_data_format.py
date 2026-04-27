import os
import json
from dotenv import load_dotenv

load_dotenv()

PROCESSED_PATH = os.getenv("DATA_PROCESSED_DIR").rstrip("/")

INPUT_FILE = f"{PROCESSED_PATH}/dpo/mbr_scored.jsonl"
OUTPUT_FILE_BW = f"{PROCESSED_PATH}/dpo/dpo_train_bw.jsonl"
OUTPUT_FILE_BMW = f"{PROCESSED_PATH}/dpo/dpo_train_bmw.jsonl"
MIN_SCORE_GAP = 0.05


def build_prompt(source: str) -> str:
    """
    Reconstruct the ChatML user turn.
    The prompt passed to DPOTrainer must NOT include the assistant turn —
    DPOTrainer appends chosen/rejected itself during training.
    """
    return (
        f"<|im_start|>user\n"
        f"Translate the following text from English into French.\n"
        f"{source}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )


def wrap_completion(text: str) -> str:
    """
    Wrap the translation as the assistant completion.
    The <|im_end|> signals end of turn to the model.
    """
    return f"{text}<|im_end|>"


records = []
with open(INPUT_FILE, "r", encoding="utf-8") as f:
    for line in f:
        records.append(json.loads(line))

print(f"Total records: {len(records)}")

bw_pairs  = []
bmw_pairs = []
skipped   = 0

for record in records:
    source   = record["source"]
    chosen   = record["chosen"]
    rejected = record["rejected"]
    middle   = record["middle"]
    gap      = record["score_gap"]

    # Skip pairs with to small quality gap
    if gap < MIN_SCORE_GAP:
        skipped += 1
        continue

    prompt = build_prompt(source)

    # BW pair = best vs worst
    bw_pairs.append({
        "prompt":   prompt,
        "chosen":   wrap_completion(chosen),
        "rejected": wrap_completion(rejected),
    })

    # BMW pairs = best vs middle AND middle vs worst
    bmw_pairs.append({
        "prompt":   prompt,
        "chosen":   wrap_completion(chosen),
        "rejected": wrap_completion(rejected),
    })
    bmw_pairs.append({
        "prompt":   prompt,
        "chosen":   wrap_completion(chosen),
        "rejected": wrap_completion(middle),
    })

print(f"Skipped (gap < {MIN_SCORE_GAP}): {skipped}")
print(f"BW pairs:  {len(bw_pairs)}")
print(f"BMW pairs: {len(bmw_pairs)}")


# Output
for path, pairs in [(OUTPUT_FILE_BW, bw_pairs), (OUTPUT_FILE_BMW, bmw_pairs)]:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for pair in pairs:
            f.write(json.dumps(pair, ensure_ascii=False) + "\n")
    print(f"Saved {len(pairs)} pairs to {path}")