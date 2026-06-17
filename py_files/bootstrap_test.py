import os
import json
import argparse
import subprocess
import tempfile
from dotenv import load_dotenv

load_dotenv()
PROC   = os.getenv("DATA_PROCESSED_DIR").rstrip("/")
MODELS = os.getenv("MODELS_DIR").rstrip("/")
PYTHON = os.path.expanduser("~/master-thesis/venv/bin/python")

REFERENCES = {
    "tico": f"{PROC}/test/test.fr",
    "emea": f"{PROC}/test_emea/test.fr",
}

COMPARISONS = [
    ("tower_sft_plain",     "tower_sft_terms",     "Tower: SFT baseline vs SFT terms"),
    ("tower_sft_terms",     "tower_dpo_terms",      "Tower: SFT terms vs DPO terms"),
    ("tower_sft_plain",     "tower_dpo_terms",      "Tower: SFT baseline vs DPO terms"),
    ("qwen_sft_plain",      "qwen_sft_terms",       "Qwen3: SFT baseline vs SFT terms"),
    ("qwen_sft_terms",      "qwen_dpo_terms",       "Qwen3: SFT terms vs DPO terms"),
    ("qwen_sft_plain",      "qwen_dpo_terms",       "Qwen3: SFT baseline vs DPO terms"),
    ("ministral_sft_plain", "ministral_sft_terms",  "Ministral: SFT baseline vs SFT terms"),
    ("ministral_sft_terms", "ministral_dpo_terms",  "Ministral: SFT terms vs DPO terms"),
    ("ministral_sft_plain", "ministral_dpo_terms",  "Ministral: SFT baseline vs DPO terms"),
]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--test", nargs="+", required=True, choices=["tico", "emea"])
    p.add_argument("--metric", default="bleu", choices=["bleu", "chrf"])
    p.add_argument("--n", type=int, default=1000)
    return p.parse_args()


def hyp_path(out_dir, sys_name):
    return f"{out_dir}/hypotheses_{sys_name}.jsonl"


def run_bootstrap(hyp_a, hyp_b, ref_path, metric, n):
    """Run sacrebleu paired bootstrap, parse JSON output."""
    cmd = [
        PYTHON, "-m", "sacrebleu",
        ref_path,
        "--input", hyp_a, hyp_b,
        "--metrics", metric,
        "--paired-bs",
        "--paired-bs-n", str(n),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)

    try:
        data = json.loads(result.stdout)
    except json.JSONDecodeError:
        return None, None, result.stdout + result.stderr

    # sacrebleu returns a list: [baseline_entry, challenger_entry]
    # baseline has p_value=null, challenger has the actual p_value
    baseline_score = None
    challenger_score = None
    p_value = None

    metric_key = "chrF2" if metric == "chrf" else metric.upper()

    for entry in data:
        scores = entry.get(metric_key, {})
        score  = scores.get("score")
        pv     = scores.get("p_value")

        if pv is None:
            # this is the baseline
            baseline_score = score
        else:
            challenger_score = score
            p_value = pv

    return p_value, (baseline_score, challenger_score), result.stdout


def sig_label(p):
    if p is None:
        return "p=? (could not parse)"
    if p < 0.001:
        return f"p={p:.4f} ***"
    if p < 0.01:
        return f"p={p:.4f} **"
    if p < 0.05:
        return f"p={p:.4f} *"
    return f"p={p:.4f} (not significant)"


def main():
    args = parse_args()

    for test_name in args.test:
        ref_path = REFERENCES[test_name]
        out_dir  = f"{MODELS}/final_eval_results/{test_name}"
        results  = []

        print(f"\n{'='*70}")
        print(f"  PAIRED BOOTSTRAP — {test_name.upper()} "
              f"({args.metric.upper()}, n={args.n})")
        print(f"{'='*70}")
        print(f"  * p<0.05  ** p<0.01  *** p<0.001")
        print(f"  Baseline (A) vs Challenger (B): is B significantly better than A?")
        print(f"{'='*70}\n")

        for sys_a, sys_b, label in COMPARISONS:
            pa = hyp_path(out_dir, sys_a)
            pb = hyp_path(out_dir, sys_b)

            if not os.path.exists(pa) or not os.path.exists(pb):
                print(f"  [{label}] missing hypothesis file — skipping.")
                continue

            p_value, scores, raw = run_bootstrap(pa, pb, ref_path, args.metric, args.n)
            label_str = sig_label(p_value)

            score_str = ""
            if scores and scores[0] is not None and scores[1] is not None:
                direction = "↑" if scores[1] > scores[0] else "↓"
                score_str = (f"  (A={scores[0]:.4f} → B={scores[1]:.4f} {direction})")

            print(f"  {label}")
            print(f"    {label_str}{score_str}")
            print()

            results.append({
                "comparison": label,
                "system_a": sys_a,
                "system_b": sys_b,
                "score_a": scores[0] if scores else None,
                "score_b": scores[1] if scores else None,
                "p_value": p_value,
                "significant": p_value is not None and p_value < 0.05,
            })

        out_path = f"{out_dir}/bootstrap_{args.metric}.json"
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to: {out_path}")


if __name__ == "__main__":
    main()