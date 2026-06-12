import os
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

PROCESSED_PATH = os.getenv("DATA_PROCESSED_DIR").rstrip("/")
MODELS_DIR = os.getenv("MODELS_DIR").rstrip("/")

COMET_REF_MODEL = "Unbabel/wmt22-comet-da"
MAX_NEW_TOKENS  = 256
BATCH_SIZE = 8
BETAS = [0.01, 0.05, 0.1, 0.5]

MODEL_CONFIGS = {
    "ministral": dict(
        merged=f"{MODELS_DIR}/SFT_Ministral_terms_merged",
        stem="DPO_Ministral_terms",
        val_file=f"{PROCESSED_PATH}/val/messages_val_terms.jsonl",
        is_qwen=False,
    ),
    "tower": dict(
        merged=f"{MODELS_DIR}/SFT_TowerInstruct_terms_merged",
        stem="DPO_TowerInstruct_terms",
        val_file=f"{PROCESSED_PATH}/val/messages_val_terms.jsonl",
        is_qwen=False,
    ),
    "qwen": dict(
        merged=f"{MODELS_DIR}/SFT_Qwen3_terms_merged",
        stem="DPO_Qwen3_terms",
        val_file=f"{PROCESSED_PATH}/val/messages_val_terms.jsonl",
        is_qwen=True,
    ),
}


def parse_args():
    p = argparse.ArgumentParser(description="DPO terms evaluation.")
    p.add_argument("--model", required=True, choices=["tower", "ministral", "qwen"])
    return p.parse_args()


def checkpoint_paths(stem: str) -> dict:
    return {
        f"dpo_beta{b}": f"{MODELS_DIR}/{stem}_beta{b}"
        for b in BETAS
    }

def load_existing_scores(scores_file: str) -> dict:
    if os.path.exists(scores_file):
        with open(scores_file) as f:
            scores = json.load(f)
        print(f"  Resuming — loaded existing scores from {scores_file}")
        return scores
    return {}


def load_existing_hypotheses(out_dir: str, ckpt_name: str):
    hyp_path = f"{out_dir}/hypotheses_{ckpt_name}.jsonl"
    if os.path.exists(hyp_path):
        hypotheses = []
        with open(hyp_path, "r", encoding="utf-8") as f:
            for line in f:
                hypotheses.append(json.loads(line)["hypothesis"])
        print(f"  Resuming — loaded {len(hypotheses)} hypotheses from {hyp_path}")
        return hypotheses
    return None


def save_hypotheses(out_dir: str, ckpt_name: str, hypotheses: list):
    os.makedirs(out_dir, exist_ok=True)
    hyp_path = f"{out_dir}/hypotheses_{ckpt_name}.jsonl"
    with open(hyp_path, "w", encoding="utf-8") as f:
        for h in hypotheses:
            f.write(json.dumps({"hypothesis": h}, ensure_ascii=False) + "\n")
    print(f"  Hypotheses saved to: {hyp_path}")


def save_scores(scores_file: str, results: dict):
    os.makedirs(os.path.dirname(scores_file), exist_ok=True)
    with open(scores_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Scores saved to: {scores_file}")


def build_prompt(user_content: str, is_qwen_or_tower: bool) -> str:
    if is_qwen_or_tower:
        return (
            f"<|im_start|>user\n{user_content}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )
    else:
        return f"<s>[INST]{user_content}[/INST]"


def load_val_data(path: str, is_qwen: bool):
    sources, references, prompts = [], [], []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            obj  = json.loads(line)
            msgs = obj["messages"]
            user_msg = next(m for m in msgs if m["role"] == "user")
            asst_msg = next(m for m in msgs if m["role"] == "assistant")

            source_text = ""
            for content_line in user_msg["content"].split("\n"):
                if content_line.startswith("English:"):
                    source_text = content_line[len("English:"):].strip()
                    break

            sources.append(source_text)
            references.append(asst_msg["content"].strip())
            prompts.append(build_prompt(user_msg["content"], is_qwen_or_tower=True))

    return sources, references, prompts


def load_model_and_tokenizer(merged_path: str, checkpoint_path: str):
    tokenizer = AutoTokenizer.from_pretrained(
        merged_path, trust_remote_code=True, padding_side="left"
    )
    tokenizer.pad_token = tokenizer.eos_token

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    base_model = AutoModelForCausalLM.from_pretrained(
        merged_path,
        quantization_config=bnb_config,
        device_map={"": 0},
        trust_remote_code=True,
    )

    print(f"  Applying DPO LoRA adapter from: {checkpoint_path}")
    model = PeftModel.from_pretrained(base_model, checkpoint_path)
    model.eval()
    return model, tokenizer


def generate_translations(model, tokenizer, prompts: list, is_qwen: bool) -> list:
    hypotheses = []
    for i in tqdm(range(0, len(prompts), BATCH_SIZE), desc="  Generating"):
        batch_prompts = prompts[i: i + BATCH_SIZE]
        inputs = tokenizer(
            batch_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        ).to(model.device)

        with torch.no_grad():
            if is_qwen:
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=MAX_NEW_TOKENS,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    enable_thinking=False, # no generation for qwen
                )
            else:
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=MAX_NEW_TOKENS,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )

        for j, output in enumerate(outputs):
            prompt_len = inputs["input_ids"].shape[1]
            generated  = output[prompt_len:]
            decoded    = tokenizer.decode(generated, skip_special_tokens=True).strip()
            hypotheses.append(decoded)

    return hypotheses


def compute_bleu(hypotheses, references):
    return round(sacrebleu.corpus_bleu(hypotheses, [references]).score, 4)


def compute_chrf(hypotheses, references):
    return round(sacrebleu.corpus_chrf(hypotheses, [references]).score, 4)


def compute_comet_da(sources, hypotheses, references):
    print(f"  Loading COMET model: {COMET_REF_MODEL}")
    comet_path  = download_model(COMET_REF_MODEL)
    comet_model = load_from_checkpoint(comet_path)
    data   = [{"src": s, "mt": h, "ref": r}
               for s, h, r in zip(sources, hypotheses, references)]
    output = comet_model.predict(data, batch_size=16, gpus=1)
    del comet_model
    torch.cuda.empty_cache()
    return round(output.system_score, 4)


def print_summary(model_key: str, results: dict):
    print("\n" + "=" * 65)
    print(f"  EVALUATION SUMMARY — DPO {model_key} terms — Validation Set")
    print("=" * 65)
    print(f"{'Model':<20} {'BLEU':>8} {'ChrF':>8} {'COMET-DA':>10}")
    print("-" * 65)
    for ckpt, scores in results.items():
        print(f"{ckpt:<20} {str(scores.get('bleu','—')):>8} "
              f"{str(scores.get('chrf','—')):>8} "
              f"{str(scores.get('comet_wmt22','—')):>10}")
    print("=" * 65)


def main():
    args   = parse_args()
    cfg    = MODEL_CONFIGS[args.model]
    merged = cfg["merged"]
    is_qwen = cfg["is_qwen"]

    out_dir     = f"{MODELS_DIR}/{cfg['stem']}_eval_results"
    scores_file = f"{out_dir}/val_scores_dpo.json"
    checkpoints = checkpoint_paths(cfg["stem"])

    missing = [p for p in checkpoints.values() if not os.path.isdir(p)]
    if missing:
        raise SystemExit(f"Missing checkpoint directories:\n" + "\n".join(missing))

    print(f"Evaluating {args.model} (terms) — {len(checkpoints)} beta runs")
    print(f"Loading validation data from: {cfg['val_file']}")
    sources, references, prompts = load_val_data(cfg["val_file"], is_qwen)
    print(f"  {len(prompts)} examples loaded.")

    results = load_existing_scores(scores_file)

    for ckpt_name, ckpt_path in checkpoints.items():
        print(f"\n{'='*50}\n  Evaluating: {ckpt_name}\n{'='*50}")
        ckpt_scores = results.get(ckpt_name, {})

        hypotheses = load_existing_hypotheses(out_dir, ckpt_name)
        if hypotheses is None:
            model, tokenizer = load_model_and_tokenizer(merged, ckpt_path)
            hypotheses = generate_translations(model, tokenizer, prompts, is_qwen)
            save_hypotheses(out_dir, ckpt_name, hypotheses)
            del model
            torch.cuda.empty_cache()
        else:
            print("  Skipping generation (cached).")

        for metric, compute_fn, kwargs in [
            ("bleu",       compute_bleu,     dict(hypotheses=hypotheses, references=references)),
            ("chrf",       compute_chrf,     dict(hypotheses=hypotheses, references=references)),
            ("comet_wmt22",compute_comet_da, dict(sources=sources, hypotheses=hypotheses, references=references)),
        ]:
            if metric not in ckpt_scores:
                print(f"  Computing {metric.upper()}...")
                ckpt_scores[metric] = compute_fn(**kwargs)
                print(f"  {metric.upper()}: {ckpt_scores[metric]}")
                results[ckpt_name] = ckpt_scores
                save_scores(scores_file, results)
            else:
                print(f"  {metric.upper()} already computed: {ckpt_scores[metric]} — skipping.")

    print_summary(args.model, results)


if __name__ == "__main__":
    main()