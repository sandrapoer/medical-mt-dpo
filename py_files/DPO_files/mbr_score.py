import os
import json
import torch
from dotenv import load_dotenv
from comet import download_model, load_from_checkpoint
from tqdm import tqdm

load_dotenv()

PROCESSED_PATH = os.getenv("DATA_PROCESSED_DIR").rstrip("/")

INPUT_FILE = f"{PROCESSED_PATH}/dpo/hypotheses.jsonl"
OUTPUT_FILE = f"{PROCESSED_PATH}/dpo/mbr_scored.jsonl"
COMET_MODEL = "Unbabel/wmt22-comet-da"
BATCH_SIZE = 64


print("Loading COMET model...")
comet_path  = download_model(COMET_MODEL)
comet_model = load_from_checkpoint(comet_path)
comet_model.eval()

# Resume logic
completed = set()
if os.path.exists(OUTPUT_FILE):
    with open(OUTPUT_FILE, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            completed.add(obj["source"])
    print(f"Resuming — {len(completed)} examples already scored.")


# Load hypotheses
records = []
with open(INPUT_FILE, "r", encoding="utf-8") as f:
    for line in f:
        obj = json.loads(line)
        if obj["source"] not in completed:
            records.append(obj)

print(f"{len(records)} examples to score.")

outfile = open(OUTPUT_FILE, "a", encoding="utf-8")

for record in tqdm(records, desc="MBR scoring"):
    source      = record["source"]
    hypotheses  = record["hypotheses"]  # list of H strings
    H           = len(hypotheses)

    comet_inputs = []
    pair_index   = []  # to track which hypothesis is being scored in each pair
    for i in range(H):
        for j in range(H):
            if i == j:
                continue
            comet_inputs.append({
                "src": source,
                "mt":  hypotheses[i],# hypothesis being scored
                "ref": hypotheses[j],# pseudo-reference
            })
            pair_index.append(i)


    all_scores = []
    for start in range(0, len(comet_inputs), BATCH_SIZE):
        batch  = comet_inputs[start : start + BATCH_SIZE]
        result = comet_model.predict(batch, batch_size=BATCH_SIZE, gpus=1)
        all_scores.extend(result.scores)


    mbr_scores = [0.0] * H
    counts     = [0]   * H
    for score, i in zip(all_scores, pair_index):
        mbr_scores[i] += score
        counts[i]     += 1
    mbr_scores = [mbr_scores[i] / counts[i] for i in range(H)]


    ranked = sorted(range(H), key=lambda i: mbr_scores[i], reverse=True)

    chosen  = hypotheses[ranked[0]]   # highest MBR score
    rejected = hypotheses[ranked[-1]] # lowest MBR score
    middle  = hypotheses[ranked[H // 2]]  # middle — for BMW strategy if needed


    score_gap = mbr_scores[ranked[0]] - mbr_scores[ranked[-1]]

    record_out = {
        "source":      source,
        "hypotheses":  hypotheses,
        "mbr_scores":  mbr_scores,
        "ranked":      ranked,
        "chosen":      chosen,
        "rejected":    rejected,
        "middle":      middle,
        "score_gap":   score_gap,
    }
    outfile.write(json.dumps(record_out, ensure_ascii=False) + "\n")

outfile.close()
print(f"Done. Scored pairs saved to {OUTPUT_FILE}")