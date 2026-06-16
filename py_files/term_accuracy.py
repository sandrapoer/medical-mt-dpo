import os
import re
import json
import argparse
from dotenv import load_dotenv

load_dotenv()
PROC   = os.getenv("DATA_PROCESSED_DIR").rstrip("/")
MODELS = os.getenv("MODELS_DIR").rstrip("/")

CACHE_PATH = f"{PROC}/dpo/umls_cache.json"

TERMS_SYSTEMS = [
    "tower_sft_terms",
    "tower_dpo_terms",
    "qwen_sft_terms",
    "qwen_dpo_terms",
    "ministral_sft_terms",
    "ministral_dpo_terms",
]

TEST_TERMS_FILES = {
    "tico": f"{PROC}/test/messages_test_terms.jsonl",
    "emea": f"{PROC}/test_emea/messages_test_terms.jsonl",
}

# Matches injected glossary pairs: "patient" -> "patient"
PAIR_RE = re.compile(r'"([^"]+)"\s*->\s*"([^"]+)"')


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--test", nargs="+", required=True, choices=["tico", "emea"])
    return p.parse_args()


def load_umls_cache(path: str) -> dict:
    """Returns {en_term: [fr_term1, fr_term2, ...]}"""
    with open(path, encoding="utf-8") as f:
        cache = json.load(f)
    # Build flat set of all known FR terms for precision denominator.
    all_fr = set()
    for fr_list in cache.values():
        for t in fr_list:
            if t:
                all_fr.add(t.lower())
    print(f"UMLS cache: {len(cache)} EN terms, {len(all_fr)} unique FR terms")
    return cache, all_fr


def load_terms_data(path: str):
    """Load only sentences that had terms injected.
    Returns list of {source, reference, injected_pairs: [(en, fr), ...]}
    """
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            obj  = json.loads(line)
            msgs = obj["messages"]
            user = next(m for m in msgs if m["role"] == "user")
            asst = next(m for m in msgs if m["role"] == "assistant")
            content = user["content"]
            if "Glossaries:" not in content:
                continue
            pairs = PAIR_RE.findall(content)
            if not pairs:
                continue
            src = ""
            for l in content.split("\n"):
                if l.startswith("English:"):
                    src = l[len("English:"):].strip()
                    break
            records.append({
                "source":         src,
                "reference":      asst["content"].strip(),
                "injected_pairs": pairs,  # [(en_term, fr_term), ...]
            })
    return records


def load_hypotheses(out_dir: str, sys_name: str):
    path = f"{out_dir}/hypotheses_{sys_name}.jsonl"
    if not os.path.exists(path):
        return None
    return [json.loads(l)["hypothesis"] for l in open(path, encoding="utf-8")]


def term_found(fr_term: str, hypothesis: str) -> bool:
    """Substring match after lowercasing — handles French morphological variants."""
    return fr_term.lower() in hypothesis.lower()


def build_src_to_hyp(test_path: str, all_hyps: list) -> dict:
    """Map source sentences to their hypothesis by position in the full file."""
    src_to_hyp = {}
    with open(test_path, encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if idx >= len(all_hyps):
                break
            obj  = json.loads(line)
            msgs = obj["messages"]
            user = next(m for m in msgs if m["role"] == "user")
            src = ""
            for l in user["content"].split("\n"):
                if l.startswith("English:"):
                    src = l[len("English:"):].strip()
                    break
            src_to_hyp[src] = all_hyps[idx]
    return src_to_hyp


def evaluate_system(sys_name, records, all_hyps, test_name, all_fr_terms):
    src_to_hyp = build_src_to_hyp(TEST_TERMS_FILES[test_name], all_hyps)

    # Accumulators
    total_injected   = 0   # denominator for recall
    recall_hits      = 0   # injected FR terms found in hyp
    precision_num    = 0   # injected FR terms found in hyp (same as recall_hits)
    precision_den    = 0   # all UMLS FR terms found in hyp
    sents_full_recall = 0

    for rec in records:
        hyp   = src_to_hyp.get(rec["source"], "")
        pairs = rec["injected_pairs"]

        # Recall: how many injected FR terms appear in hypothesis
        n_hits = sum(1 for _, fr in pairs if term_found(fr, hyp))
        recall_hits    += n_hits
        total_injected += len(pairs)
        if n_hits == len(pairs):
            sents_full_recall += 1

        # Precision denominator: count all UMLS FR terms found in hypothesis
        umls_in_hyp = sum(1 for fr in all_fr_terms if term_found(fr, hyp))
        precision_num += n_hits
        precision_den += umls_in_hyp

    recall    = recall_hits / total_injected if total_injected > 0 else 0.0
    precision = precision_num / precision_den if precision_den > 0 else 0.0
    f1        = (2 * precision * recall / (precision + recall)
                 if (precision + recall) > 0 else 0.0)
    sent_rec  = sents_full_recall / len(records) if records else 0.0

    return {
        "term_recall":          round(recall * 100, 2),
        "term_precision":       round(precision * 100, 2),
        "term_f1":              round(f1 * 100, 2),
        "sentence_recall":      round(sent_rec * 100, 2),
        "total_injected_terms": total_injected,
        "recall_hits":          recall_hits,
        "sents_evaluated":      len(records),
    }


def print_table(test_name: str, all_results: dict):
    print("\n" + "=" * 76)
    print(f"  TERMINOLOGY ACCURACY — {test_name.upper()}")
    print("  Recall   : injected FR terms found in hypothesis / total injected")
    print("  Precision: injected FR terms found / all UMLS FR terms in hypothesis")
    print("  F1       : harmonic mean of precision and recall")
    print("=" * 76)
    print(f"{'System':<28} {'Recall%':>9} {'Prec%':>7} {'F1%':>6} {'N sents':>8}")
    print("-" * 76)
    for sys_name, r in all_results.items():
        print(f"{sys_name:<28} "
              f"{r['term_recall']:>9.2f} "
              f"{r['term_precision']:>7.2f} "
              f"{r['term_f1']:>6.2f} "
              f"{r['sents_evaluated']:>8}")
    print("=" * 76)


def main():
    args = parse_args()

    print("Loading UMLS cache...")
    _, all_fr_terms = load_umls_cache(CACHE_PATH)

    for test_name in args.test:
        print(f"\n--- {test_name.upper()} ---")
        records = load_terms_data(TEST_TERMS_FILES[test_name])
        print(f"  Sentences with injected terms: {len(records)}")
        print(f"  Total injected term pairs: "
              f"{sum(len(r['injected_pairs']) for r in records)}")

        out_dir    = f"{MODELS}/final_eval_results/{test_name}"
        all_results = {}

        for sys_name in TERMS_SYSTEMS:
            hyps = load_hypotheses(out_dir, sys_name)
            if hyps is None:
                print(f"  [{sys_name}] no hypotheses yet — skipping.")
                continue
            result = evaluate_system(
                sys_name, records, hyps, test_name, all_fr_terms
            )
            all_results[sys_name] = result
            print(f"  [{sys_name}] "
                  f"recall={result['term_recall']}% "
                  f"precision={result['term_precision']}% "
                  f"F1={result['term_f1']}%")

        if all_results:
            print_table(test_name, all_results)
            out_path = f"{out_dir}/term_accuracy.json"
            with open(out_path, "w") as f:
                json.dump(all_results, f, indent=2)
            print(f"Saved to: {out_path}")
        else:
            print("  No hypothesis files found yet. Run final_evaluation.py first.")


if __name__ == "__main__":
    main()