"""Unified final test evaluation — TICO-19 + EMEA in-domain.

Evaluates all valid systems on both test sets:
  SFT plain      : Tower, Qwen3, Ministral (negative baseline)
  SFT terms      : Tower, Qwen3, Ministral (negative baseline)
  DPO terms β=0.01: Tower, Qwen3, Ministral (negative baseline)

Only the best SFT checkpoint per model is evaluated (determined from val set).
DPO systems load the merged SFT base + DPO LoRA adapter at β=0.01 root dir.

Usage
-----
    # TICO-19 (out-of-domain)
    CUDA_VISIBLE_DEVICES=1 python py_files/final_evaluation.py --test tico

    # EMEA (in-domain)
    CUDA_VISIBLE_DEVICES=2 python py_files/final_evaluation.py --test emea

Run one test set at a time — each takes several hours due to COMET.
Results saved with resume logic so crashes are recoverable.
"""

import os
import gc
import json
import argparse

import torch
import sacrebleu
from dotenv import load_dotenv
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
from comet import download_model, load_from_checkpoint

load_dotenv()
PROC      = os.getenv("DATA_PROCESSED_DIR").rstrip("/")
MODELS    = os.getenv("MODELS_DIR").rstrip("/")

COMET_MODEL  = "Unbabel/wmt22-comet-da"
MAX_NEW_TOKENS = 256
BATCH_SIZE     = 8

# --------------------------------------------------------------------------- #
# System registry
# Each entry: base model HF name, adapter path, merged path (for DPO),
# prompt style, whether to suppress Qwen3 thinking.
# --------------------------------------------------------------------------- #
SYSTEMS = {
    # ---- Tower ----
    "tower_sft_plain": dict(
        base="Unbabel/TowerInstruct-7B-v0.2",
        adapter=f"{MODELS}/SFT_TowerInstruct_final/checkpoint-1250",
        merged=None,
        style="chatml", is_qwen=False,
        data="plain",
    ),
    "tower_sft_terms": dict(
        base="Unbabel/TowerInstruct-7B-v0.2",
        adapter=f"{MODELS}/SFT_TowerInstruct_terms_merged",   # merged adapter
        merged=None,
        style="chatml", is_qwen=False,
        data="terms",
    ),
    "tower_dpo_terms": dict(
        base=None,
        adapter=f"{MODELS}/DPO_TowerInstruct_terms_beta0.01",
        merged=f"{MODELS}/SFT_TowerInstruct_terms_merged",
        style="chatml", is_qwen=False,
        data="terms",
    ),
    # ---- Qwen3 ----
    "qwen_sft_plain": dict(
        base="Qwen/Qwen3-8B",
        adapter=f"{MODELS}/SFT_Qwen3_final/checkpoint-1250",
        merged=None,
        style="chatml", is_qwen=True,
        data="plain",
    ),
    "qwen_sft_terms": dict(
        base="Qwen/Qwen3-8B",
        adapter=f"{MODELS}/SFT_Qwen3_terms/checkpoint-1875",
        merged=None,
        style="chatml", is_qwen=True,
        data="terms",
    ),
    "qwen_dpo_terms": dict(
        base=None,
        adapter=f"{MODELS}/DPO_Qwen3_terms_beta0.01",
        merged=f"{MODELS}/SFT_Qwen3_terms_merged",
        style="chatml", is_qwen=True,
        data="terms",
    ),
    # ---- Ministral (negative baseline) ----
    "ministral_sft_plain": dict(
        base="mistralai/Ministral-8B-Instruct-2410",
        adapter=f"{MODELS}/SFT_Ministral_final/checkpoint-1875",
        merged=None,
        style="ministral", is_qwen=False,
        data="plain",
    ),
    "ministral_sft_terms": dict(
        base="mistralai/Ministral-8B-Instruct-2410",
        adapter=f"{MODELS}/SFT_Ministral_terms/checkpoint-1875",
        merged=None,
        style="ministral", is_qwen=False,
        data="terms",
    ),
    "ministral_dpo_terms": dict(
        base=None,
        adapter=f"{MODELS}/DPO_Ministral_terms_beta0.01",
        merged=f"{MODELS}/SFT_Ministral_terms_merged",
        style="ministral", is_qwen=False,
        data="terms",
    ),
}

# --------------------------------------------------------------------------- #
# Test set paths
# --------------------------------------------------------------------------- #
TEST_FILES = {
    "tico": {
        "plain": f"{PROC}/test/messages_test.jsonl",
        "terms": f"{PROC}/test/messages_test_terms.jsonl",
    },
    "emea": {
        "plain": f"{PROC}/test_emea/messages_test.jsonl",
        "terms": f"{PROC}/test_emea/messages_test_terms.jsonl",
    },
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--test", required=True, choices=["tico", "emea"])
    p.add_argument("--systems", nargs="+", default=None,
                   help="Subset of system keys to run (default: all)")
    return p.parse_args()


# --------------------------------------------------------------------------- #
# Data loading
# --------------------------------------------------------------------------- #
def load_test_data(path: str):
    sources, references, prompts = [], [], []
    with open(path, encoding="utf-8") as f:
        for line in f:
            obj  = json.loads(line)
            msgs = obj["messages"]
            user = next(m for m in msgs if m["role"] == "user")
            asst = next(m for m in msgs if m["role"] == "assistant")
            src  = ""
            for l in user["content"].split("\n"):
                if l.startswith("English:"):
                    src = l[len("English:"):].strip()
                    break
            sources.append(src)
            references.append(asst["content"].strip())
            prompts.append(user["content"])
    return sources, references, prompts


def build_prompt(content: str, style: str) -> str:
    if style == "ministral":
        return f"<s>[INST]{content}[/INST]"
    return f"<|im_start|>user\n{content}<|im_end|>\n<|im_start|>assistant\n"


# --------------------------------------------------------------------------- #
# Model loading
# --------------------------------------------------------------------------- #
def bnb():
    return BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16,
    )


def load_system(sys_cfg: dict):
    is_dpo = sys_cfg["merged"] is not None
    base_path = sys_cfg["merged"] if is_dpo else sys_cfg["base"]

    tok = AutoTokenizer.from_pretrained(
        base_path, trust_remote_code=True, padding_side="left"
    )
    tok.pad_token = tok.eos_token

    base = AutoModelForCausalLM.from_pretrained(
        base_path, quantization_config=bnb(),
        device_map={"": 0}, trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(base, sys_cfg["adapter"])
    model.eval()
    return model, tok


# --------------------------------------------------------------------------- #
# Generation
# --------------------------------------------------------------------------- #
def generate(model, tokenizer, prompts, style, is_qwen):
    hypotheses = []
    formatted  = [build_prompt(p, style) for p in prompts]
    for i in tqdm(range(0, len(formatted), BATCH_SIZE), desc="  Generating"):
        batch = formatted[i: i + BATCH_SIZE]
        inputs = tokenizer(
            batch, return_tensors="pt", padding=True,
            truncation=True, max_length=512,
        ).to(model.device)
        kw = dict(
            max_new_tokens=MAX_NEW_TOKENS, do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
        if is_qwen:
            kw["enable_thinking"] = False
        with torch.no_grad():
            outputs = model.generate(**inputs, **kw)
        for out in outputs:
            gen = out[inputs["input_ids"].shape[1]:]
            hypotheses.append(
                tokenizer.decode(gen, skip_special_tokens=True).strip()
            )
    return hypotheses


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def bleu(hyps, refs):
    return round(sacrebleu.corpus_bleu(hyps, [refs]).score, 4)

def chrf(hyps, refs):
    return round(sacrebleu.corpus_chrf(hyps, [refs]).score, 4)

def comet_da(srcs, hyps, refs):
    path  = download_model(COMET_MODEL)
    mdl   = load_from_checkpoint(path)
    data  = [{"src": s, "mt": h, "ref": r} for s, h, r in zip(srcs, hyps, refs)]
    score = mdl.predict(data, batch_size=16, gpus=1).system_score
    del mdl; torch.cuda.empty_cache()
    return round(score, 4)


# --------------------------------------------------------------------------- #
# I/O helpers
# --------------------------------------------------------------------------- #
def hyp_path(out_dir, sys_name):
    return f"{out_dir}/hypotheses_{sys_name}.jsonl"

def load_hyps(out_dir, sys_name):
    p = hyp_path(out_dir, sys_name)
    if not os.path.exists(p):
        return None
    return [json.loads(l)["hypothesis"] for l in open(p, encoding="utf-8")]

def save_hyps(out_dir, sys_name, hyps):
    os.makedirs(out_dir, exist_ok=True)
    with open(hyp_path(out_dir, sys_name), "w", encoding="utf-8") as f:
        for h in hyps:
            f.write(json.dumps({"hypothesis": h}, ensure_ascii=False) + "\n")

def load_scores(scores_file):
    if os.path.exists(scores_file):
        with open(scores_file) as f:
            return json.load(f)
    return {}

def save_scores(scores_file, results):
    os.makedirs(os.path.dirname(scores_file), exist_ok=True)
    with open(scores_file, "w") as f:
        json.dump(results, f, indent=2)


# --------------------------------------------------------------------------- #
# Summary
# --------------------------------------------------------------------------- #
def print_summary(test_name, results):
    print("\n" + "=" * 70)
    print(f"  FINAL TEST RESULTS — {test_name.upper()}")
    print("=" * 70)
    print(f"{'System':<28} {'BLEU':>8} {'ChrF':>8} {'COMET-DA':>10}")
    print("-" * 70)
    for sys_name, scores in results.items():
        print(f"{sys_name:<28} "
              f"{str(scores.get('bleu','—')):>8} "
              f"{str(scores.get('chrf','—')):>8} "
              f"{str(scores.get('comet_wmt22','—')):>10}")
    print("=" * 70)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    args     = parse_args()
    out_dir  = f"{MODELS}/final_eval_results/{args.test}"
    scores_f = f"{out_dir}/test_scores.json"
    results  = load_scores(scores_f)

    systems_to_run = args.systems or list(SYSTEMS.keys())

    for sys_name in systems_to_run:
        if sys_name not in SYSTEMS:
            print(f"Unknown system {sys_name!r} — skipping.")
            continue
        cfg      = SYSTEMS[sys_name]
        data_key = cfg["data"]            # "plain" or "terms"
        data_f   = TEST_FILES[args.test][data_key]

        if not os.path.isfile(data_f):
            print(f"\n[{sys_name}] test file not found: {data_f} — skipping.")
            print("  (Run make_tico_messages.py or term_instructions.py first.)")
            continue

        print(f"\n{'='*60}\n  {sys_name}  [{args.test}]\n{'='*60}")
        sys_scores = results.get(sys_name, {})

        # Generation
        hyps = load_hyps(out_dir, sys_name)
        if hyps is None:
            srcs, refs, prompts = load_test_data(data_f)
            model, tok = load_system(cfg)
            hyps = generate(model, tok, prompts, cfg["style"], cfg["is_qwen"])
            save_hyps(out_dir, sys_name, hyps)
            del model; gc.collect(); torch.cuda.empty_cache()
        else:
            srcs, refs, _ = load_test_data(data_f)
            print("  Loaded cached hypotheses.")

        # Metrics
        for metric, fn, kw in [
            ("bleu",        bleu,     dict(hyps=hyps, refs=refs)),
            ("chrf",        chrf,     dict(hyps=hyps, refs=refs)),
            ("comet_wmt22", comet_da, dict(srcs=srcs, hyps=hyps, refs=refs)),
        ]:
            if metric not in sys_scores:
                print(f"  Computing {metric.upper()}...")
                sys_scores[metric] = fn(**kw)
                print(f"  {metric.upper()}: {sys_scores[metric]}")
                results[sys_name] = sys_scores
                save_scores(scores_f, results)
            else:
                print(f"  {metric.upper()} cached: {sys_scores[metric]}")

    print_summary(args.test, results)
    print(f"\nFull results saved to: {scores_f}")


if __name__ == "__main__":
    main()