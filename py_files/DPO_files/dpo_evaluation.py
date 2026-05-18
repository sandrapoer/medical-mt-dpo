import os
import json
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

# DPO models load from merged SFT base + DPO LoRA adapter
MERGED_MODEL = f"{MODELS_DIR}/SFT_TowerInstruct_merged"

VAL_FILE = f"{PROCESSED_PATH}/val/messages_val.jsonl"
OUT_DIR = f"{MODELS_DIR}/DPO_TowerInstruct_eval_results"
SCORES_FILE = f"{OUT_DIR}/val_scores_dpo.json"

CHECKPOINTS = {
    "dpo_beta0.01": f"{MODELS_DIR}/DPO_TowerInstruct_beta0.01/checkpoint-1000",
    "dpo_beta0.05": f"{MODELS_DIR}/DPO_TowerInstruct_beta0.05/checkpoint-1000",
    "dpo_beta0.1": f"{MODELS_DIR}/DPO_TowerInstruct_beta0.1/checkpoint-1000",
    "dpo_beta0.5": f"{MODELS_DIR}/DPO_TowerInstruct_beta0.5/checkpoint-1000",
}

COMET_REF_MODEL = "Unbabel/wmt22-comet-da"

MAX_NEW_TOKENS = 256
BATCH_SIZE     = 8


def load_existing_scores() -> dict:
    if os.path.exists(SCORES_FILE):
        with open(SCORES_FILE) as f:
            scores = json.load(f)
        print(f"  Resuming — loaded existing scores from {SCORES_FILE}")
        return scores
    return {}


def load_existing_hypotheses(ckpt_name: str):
    hyp_path = f"{OUT_DIR}/hypotheses_{ckpt_name}.jsonl"
    if os.path.exists(hyp_path):
        hypotheses = []
        with open(hyp_path, "r", encoding="utf-8") as f:
            for line in f:
                hypotheses.append(json.loads(line)["hypothesis"])
        print(f"  Resuming — loaded {len(hypotheses)} hypotheses from {hyp_path}")
        return hypotheses
    return None


def save_hypotheses(ckpt_name: str, hypotheses: list):
    os.makedirs(OUT_DIR, exist_ok=True)
    hyp_path = f"{OUT_DIR}/hypotheses_{ckpt_name}.jsonl"
    with open(hyp_path, "w", encoding="utf-8") as f:
        for h in hypotheses:
            f.write(json.dumps({"hypothesis": h}, ensure_ascii=False) + "\n")
    print(f"  Hypotheses saved to: {hyp_path}")


def save_scores(results: dict):
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(SCORES_FILE, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Scores saved to: {SCORES_FILE}")


def load_val_data(path: str):
    sources, references, prompts = [], [], []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            msgs = obj["messages"]

            user_msg = next(m for m in msgs if m["role"] == "user")
            asst_msg = next(m for m in msgs if m["role"] == "assistant")

            source_text = ""
            for content_line in user_msg["content"].split("\n"):
                if content_line.startswith("English:"):
                    source_text = content_line[len("English:"):].strip()
                    break

            prompt = (
                f"<|im_start|>user\n{user_msg['content']}<|im_end|>\n"
                f"<|im_start|>assistant\n"
            )

            sources.append(source_text)
            references.append(asst_msg["content"].strip())
            prompts.append(prompt)

    return sources, references, prompts


def load_model_and_tokenizer(checkpoint_path: str):
    # load merged SFT base, then apply DPO LoRA adapter
    print(f"  Loading tokenizer from merged model: {MERGED_MODEL}")
    tokenizer = AutoTokenizer.from_pretrained(MERGED_MODEL, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    print(f"  Loading merged SFT base model...")
    base_model = AutoModelForCausalLM.from_pretrained(
        MERGED_MODEL,
        quantization_config=bnb_config,
        device_map={"": 0},
        trust_remote_code=True,
    )

    print(f"  Applying DPO LoRA adapter from: {checkpoint_path}")
    model = PeftModel.from_pretrained(base_model, checkpoint_path)
    model.eval()
    return model, tokenizer


def generate_translations(model, tokenizer, prompts: list) -> list:
    hypotheses = []
    for i in tqdm(range(0, len(prompts), BATCH_SIZE), desc="  Generating"):
        batch_prompts = prompts[i : i + BATCH_SIZE]
        inputs = tokenizer(
            batch_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        ).to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        for output in outputs:
            prompt_len = inputs["input_ids"].shape[1]
            generated = output[prompt_len:]
            decoded = tokenizer.decode(generated, skip_special_tokens=True).strip()
            hypotheses.append(decoded)

    return hypotheses


def compute_bleu(hypotheses: list, references: list) -> float:
    result = sacrebleu.corpus_bleu(hypotheses, [references])
    return round(result.score, 4)


def compute_chrf(hypotheses: list, references: list) -> float:
    result = sacrebleu.corpus_chrf(hypotheses, [references])
    return round(result.score, 4)


def compute_comet_da(sources: list, hypotheses: list, references: list) -> float:
    print(f"  Loading COMET model: {COMET_REF_MODEL}")
    comet_path  = download_model(COMET_REF_MODEL)
    comet_model = load_from_checkpoint(comet_path)

    data = [{"src": s, "mt": h, "ref": r}
            for s, h, r in zip(sources, hypotheses, references)]

    output = comet_model.predict(data, batch_size=16, gpus=1)

    del comet_model
    torch.cuda.empty_cache()

    return round(output.system_score, 4)


def print_summary(results: dict):
    print("\n" + "=" * 65)
    print("  EVALUATION SUMMARY — DPO TowerInstruct — Validation Set")
    print("=" * 65)
    print(f"{'Model':<20} {'BLEU':>8} {'ChrF':>8} {'COMET-DA':>10}")
    print("-" * 65)
    for ckpt, scores in results.items():
        bleu     = scores.get("bleu", "—")
        chrf     = scores.get("chrf", "—")
        comet_da = scores.get("comet_wmt22", "—")
        print(f"{ckpt:<20} {str(bleu):>8} {str(chrf):>8} {str(comet_da):>10}")
    print("=" * 65)
    print("\nNote: Paired bootstrap significance testing (BLEU + ChrF)")
    print("to be run via sacrebleu CLI after all models are evaluated.")


def main():
    print("Loading validation data...")
    sources, references, prompts = load_val_data(VAL_FILE)
    print(f"  {len(prompts)} examples loaded.")

    results = load_existing_scores()

    for ckpt_name, ckpt_path in CHECKPOINTS.items():
        print(f"\n{'='*50}")
        print(f"  Evaluating: {ckpt_name}")
        print(f"{'='*50}")

        ckpt_scores = results.get(ckpt_name, {})

        hypotheses = load_existing_hypotheses(ckpt_name)
        if hypotheses is None:
            model, tokenizer = load_model_and_tokenizer(ckpt_path)
            hypotheses = generate_translations(model, tokenizer, prompts)
            save_hypotheses(ckpt_name, hypotheses)
            del model
            torch.cuda.empty_cache()
        else:
            print("  Skipping generation.")

        # BLEU
        if "bleu" not in ckpt_scores:
            print("  Computing BLEU...")
            ckpt_scores["bleu"] = compute_bleu(hypotheses, references)
            print(f"  BLEU: {ckpt_scores['bleu']}")
            results[ckpt_name] = ckpt_scores
            save_scores(results)
        else:
            print(f"  BLEU already computed: {ckpt_scores['bleu']} — skipping.")

        # ChrF
        if "chrf" not in ckpt_scores:
            print("  Computing ChrF...")
            ckpt_scores["chrf"] = compute_chrf(hypotheses, references)
            print(f"  ChrF: {ckpt_scores['chrf']}")
            results[ckpt_name] = ckpt_scores
            save_scores(results)
        else:
            print(f"  ChrF already computed: {ckpt_scores['chrf']} — skipping.")

        # COMET-DA
        if "comet_wmt22" not in ckpt_scores:
            print("  Computing COMET-DA (wmt22-comet-da)...")
            ckpt_scores["comet_wmt22"] = compute_comet_da(sources, hypotheses, references)
            print(f"  COMET-DA: {ckpt_scores['comet_wmt22']}")
            results[ckpt_name] = ckpt_scores
            save_scores(results)
        else:
            print(f"  COMET-DA already computed: {ckpt_scores['comet_wmt22']} — skipping.")

    print_summary(results)


if __name__ == "__main__":
    main()