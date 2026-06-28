"""
Finds high-COMET / low-BLEU segments in the EMEA terms hypothesis files
where the injected glossary term is absent from the hypothesis (synonym substitution).
Reads existing hypothesis files only — does NOT load any translation model.
"""
import json
import re
import sacrebleu
import torch
from comet import download_model, load_from_checkpoint
from dotenv import load_dotenv
import os

load_dotenv()
PROC   = os.getenv("DATA_PROCESSED_DIR").rstrip("/")
MODELS = os.getenv("MODELS_DIR").rstrip("/")

TEST_TERMS = f"{PROC}/test_emea/messages_test_terms.jsonl"

SYSTEMS = {
    "tower_sft_terms":   f"{MODELS}/final_eval_results/emea/hypotheses_tower_sft_terms.jsonl",
    "ministral_dpo_terms": f"{MODELS}/final_eval_results/emea/hypotheses_ministral_dpo_terms.jsonl",
    "qwen_dpo_terms":    f"{MODELS}/final_eval_results/emea/hypotheses_qwen_dpo_terms.jsonl",
}

GLOSSARY_RE = re.compile(r'"([^"]+)"\s*->\s*"([^"]+)"')
SOURCE_RE   = re.compile(r'English:\s*(.+)', re.DOTALL)


def parse_test_file(path):
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            obj  = json.loads(line)
            msgs = obj["messages"]
            user = next(m for m in msgs if m["role"] == "user")
            asst = next(m for m in msgs if m["role"] == "assistant")
            content = user["content"]

            # Extract all injected glossary pairs
            glossary = GLOSSARY_RE.findall(content)  # [(en, fr), ...]

            # Extract English source
            src_match = SOURCE_RE.search(content)
            source = src_match.group(1).strip() if src_match else ""
            # Strip trailing "French:" prompt line if present
            source = re.sub(r'\s*French:\s*$', '', source).strip()

            reference = asst["content"].strip()
            records.append({"source": source, "reference": reference, "glossary": glossary})
    return records


def load_hypotheses(path):
    hyps = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            hyps.append(json.loads(line)["hypothesis"])
    return hyps


def sentence_bleu(hyp, ref):
    return sacrebleu.sentence_bleu(hyp, [ref]).score


def main():
    print("Parsing test data...")
    records = parse_test_file(TEST_TERMS)
    sources    = [r["source"]    for r in records]
    references = [r["reference"] for r in records]
    glossaries = [r["glossary"]  for r in records]
    N = len(records)
    print(f"  {N} segments loaded.")

    print("\nLoading COMET model (wmt22-comet-da)...")
    comet_path  = download_model("Unbabel/wmt22-comet-da")
    comet_model = load_from_checkpoint(comet_path)
    print("  COMET model loaded.")

    for sys_name, hyp_path in SYSTEMS.items():
        print(f"\n{'='*65}")
        print(f"  System: {sys_name}")
        print(f"{'='*65}")

        hypotheses = load_hypotheses(hyp_path)
        assert len(hypotheses) == N, f"Length mismatch: {len(hypotheses)} vs {N}"

        # --- Sentence-level BLEU (CPU, fast) ---
        print("  Computing sentence-level BLEU...")
        seg_bleu = [sentence_bleu(h, r) for h, r in zip(hypotheses, references)]

        # --- Segment-level COMET (GPU) ---
        print("  Computing segment-level COMET (GPU 1)...")
        import time
        t0 = time.time()
        comet_data = [{"src": s, "mt": h, "ref": r}
                      for s, h, r in zip(sources, hypotheses, references)]
        comet_out  = comet_model.predict(comet_data, batch_size=32, gpus=1)
        elapsed    = time.time() - t0
        seg_comet  = comet_out.scores  # list of per-segment floats
        print(f"  COMET done in {elapsed:.1f}s. Corpus score: {comet_out.system_score:.4f}")

        mean_bleu  = sum(seg_bleu)  / N
        mean_comet = sum(seg_comet) / N
        print(f"  Mean sentence-BLEU : {mean_bleu:.4f}")
        print(f"  Mean segment-COMET : {mean_comet:.4f}")

        # --- Find candidates: high COMET (top tercile) + low BLEU (bottom tercile) ---
        sorted_comet = sorted(seg_comet)
        sorted_bleu  = sorted(seg_bleu)
        comet_thresh = sorted_comet[int(N * 2 / 3)]   # top third
        bleu_thresh  = sorted_bleu[int(N * 1 / 3)]    # bottom third

        candidates = []
        for i, (hyp, ref, src, glos, sbleu, scomet) in enumerate(
                zip(hypotheses, references, sources, glossaries, seg_bleu, seg_comet)):
            if scomet >= comet_thresh and sbleu <= bleu_thresh and glos:
                # Check whether any injected FR term is absent from hypothesis
                missing = [(en, fr) for en, fr in glos
                           if fr.lower() not in hyp.lower()]
                if missing:
                    candidates.append({
                        "line": i + 1,
                        "source": src,
                        "reference": ref,
                        "hypothesis": hyp,
                        "glossary": glos,
                        "missing_in_hyp": missing,
                        "seg_bleu":  round(sbleu, 4),
                        "seg_comet": round(scomet, 4),
                    })

        print(f"\n  Candidates (high-COMET, low-BLEU, glossary term missing from hyp): {len(candidates)}")

        if not candidates:
            print("  No clean candidates found for this system.")
            continue

        # Sort by COMET desc, then BLEU asc, to surface the most extreme case
        candidates.sort(key=lambda x: (-x["seg_comet"], x["seg_bleu"]))

        print(f"\n  TOP 5 CANDIDATES:")
        for c in candidates[:5]:
            print(f"\n  Line {c['line']} | BLEU={c['seg_bleu']} | COMET={c['seg_comet']}")
            print(f"  Glossary : {c['glossary']}")
            print(f"  Missing  : {c['missing_in_hyp']}")
            print(f"  Source   : {c['source'][:120]}")
            print(f"  Reference: {c['reference'][:120]}")
            print(f"  Hypothesis:{c['hypothesis'][:120]}")

        # Print the top candidate in full
        best = candidates[0]
        print(f"\n{'='*65}")
        print(f"  BEST CANDIDATE — {sys_name}")
        print(f"{'='*65}")
        print(f"  File      : {hyp_path}")
        print(f"  Line      : {best['line']}")
        print(f"  Seg BLEU  : {best['seg_bleu']}  (corpus mean: {mean_bleu:.4f})")
        print(f"  Seg COMET : {best['seg_comet']}  (corpus mean: {mean_comet:.4f})")
        print(f"  Glossary  : {best['glossary']}")
        print(f"  Missing   : {best['missing_in_hyp']}")
        print(f"\n  SOURCE    : {best['source']}")
        print(f"\n  REFERENCE : {best['reference']}")
        print(f"\n  HYPOTHESIS: {best['hypothesis']}")

    del comet_model
    torch.cuda.empty_cache()
    print("\nDone.")


if __name__ == "__main__":
    main()
