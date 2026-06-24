import os
import gc
import json
import torch
import sacrebleu
from dotenv import load_dotenv
from tqdm import tqdm
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, PeftModel
from comet import download_model, load_from_checkpoint

load_dotenv()

PROCESSED_PATH = os.getenv("DATA_PROCESSED_DIR").rstrip("/")
MODEL_PATH = os.getenv("MODELS_DIR").rstrip("/")

MODEL_NAME = "Qwen/Qwen3-8B"
OUTPUT_DIR = f"{MODEL_PATH}/SFT_Qwen3_additional_test"

EVAL_OUT_DIR  = f"{MODEL_PATH}/final_eval_results/emea"
SCORED_JSONL  = f"{EVAL_OUT_DIR}/additional_test_qwen_sft_scored.jsonl"
SUMMARY_JSON  = f"{EVAL_OUT_DIR}/additional_test_scores.json"
COMET_MODEL   = "Unbabel/wmt22-comet-da"
EVAL_MAX_NEW  = 256
EVAL_BATCH    = 8

# ── Tokenizer ─────────────────────────────────────────────────────────────────
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME, padding_side="left", trust_remote_code=True
)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# ── 4-bit QLoRA base model ────────────────────────────────────────────────────
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map={"": 0},   # CUDA_VISIBLE_DEVICES selects the physical GPU
    trust_remote_code=True,
)
model = prepare_model_for_kbit_training(model)

# ── Additional LoRA config ─────────────────────────────────────────────────────
lora_config = LoraConfig(
    r=64,
    lora_alpha=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# ── Dataset — same path as existing SFT_qwen.py baseline ─────────────────────
dataset = load_dataset(
    "json",
    data_files={
        "train": f"{PROCESSED_PATH}/train/messages_train.jsonl",
        "validation": f"{PROCESSED_PATH}/val/messages_val.jsonl",
    },
)


def tokenize(example):
    messages = example["messages"]
    user_msg = next(m for m in messages if m["role"] == "user")
    asst_msg = next(m for m in messages if m["role"] == "assistant")

    text = (
        f"<|im_start|>user\n{user_msg['content']}<|im_end|>\n"
        f"<|im_start|>assistant\n<think>\n</think>\n"
        f"{asst_msg['content']}<|im_end|>"
    )
    return tokenizer(text, truncation=True, max_length=512)


tokenized = dataset.map(tokenize, remove_columns=dataset["train"].column_names)

# Subsample to 1000 training examples for a fast run
tokenized["train"] = tokenized["train"].select(range(1000))


class CompletionOnlyCollator:
    def __init__(self, tokenizer, response_template="<|im_start|>assistant\n"):
        self.tokenizer = tokenizer
        self.response_token_ids = tokenizer.encode(
            response_template, add_special_tokens=False
        )
        self.pad_token_id = tokenizer.pad_token_id

    def __call__(self, batch):
        input_ids_list = [torch.tensor(x["input_ids"]) for x in batch]
        max_len = max(t.size(0) for t in input_ids_list)

        input_ids = torch.stack([
            torch.nn.functional.pad(
                t, (max_len - t.size(0), 0), value=self.pad_token_id
            )
            for t in input_ids_list
        ])

        attention_mask = (input_ids != self.pad_token_id).long()
        labels = input_ids.clone()

        for i, label_seq in enumerate(labels):
            seq = label_seq.tolist()
            response_start = None
            for j in range(len(seq) - len(self.response_token_ids) + 1):
                if seq[j : j + len(self.response_token_ids)] == self.response_token_ids:
                    response_start = j + len(self.response_token_ids)
                    break
            if response_start is not None:
                labels[i, :response_start] = -100
            else:
                labels[i, :] = -100

        labels[input_ids == self.pad_token_id] = -100

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


collator = CompletionOnlyCollator(tokenizer)

# ── Training arguments ────────────────────────────────────────────────────────
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=2,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=2,       # effective batch = 4
    eval_accumulation_steps=4,
    learning_rate=2e-5,
    warmup_ratio=0.03,                   # ~300 warmup steps over 10k total
    lr_scheduler_type="cosine",
    bf16=True,
    logging_steps=10,
    eval_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=3,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    report_to="none",
    disable_tqdm=False,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized["train"],
    eval_dataset=tokenized["validation"],
    processing_class=tokenizer,
    data_collator=collator,
)

print("Starting training...")
trainer.train()
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print(f"Model and tokenizer saved to {OUTPUT_DIR}")

# ── Free training model before evaluation ────────────────────────────────────
del model, trainer
gc.collect()
torch.cuda.empty_cache()

# ─────────────────────────────────────────────────────────────────────────────
# EVALUATION on EMEA test set
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("  EVALUATION — EMEA test set")
print("=" * 60)

TEST_FILE = f"{PROCESSED_PATH}/test_emea/messages_test.jsonl"


def load_test_data(path):
    sources, references, prompts = [], [], []
    with open(path, encoding="utf-8") as f:
        for line in f:
            obj  = json.loads(line)
            msgs = obj["messages"]
            user = next(m for m in msgs if m["role"] == "user")
            asst = next(m for m in msgs if m["role"] == "assistant")
            src = ""
            for l in user["content"].split("\n"):
                if l.startswith("English:"):
                    src = l[len("English:"):].strip()
                    break
            sources.append(src)
            references.append(asst["content"].strip())
            # Qwen3 prompt with empty think block (no-thinking mode)
            prompts.append(
                f"<|im_start|>user\n{user['content']}<|im_end|>\n"
                f"<|im_start|>assistant\n<think>\n</think>\n"
            )
    return sources, references, prompts


print("Loading EMEA test data...")
sources, references, prompts = load_test_data(TEST_FILE)
print(f"  {len(prompts)} test sentences loaded.")

print("Loading trained model for evaluation...")
eval_tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME, trust_remote_code=True, padding_side="left"
)
eval_tokenizer.pad_token = eval_tokenizer.eos_token

eval_bnb = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
)
base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=eval_bnb,
    device_map={"": 0},
    trust_remote_code=True,
)
eval_model = PeftModel.from_pretrained(base_model, OUTPUT_DIR)
eval_model.eval()

print("Generating translations...")
hypotheses = []
for i in tqdm(range(0, len(prompts), EVAL_BATCH), desc="  Generating"):
    batch = prompts[i : i + EVAL_BATCH]
    inputs = eval_tokenizer(
        batch,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512,
    ).to(eval_model.device)

    with torch.no_grad():
        outputs = eval_model.generate(
            **inputs,
            max_new_tokens=EVAL_MAX_NEW,
            do_sample=False,
            top_p=0.9,          # additional-specified; no effect with do_sample=False
            pad_token_id=eval_tokenizer.pad_token_id,
            eos_token_id=eval_tokenizer.eos_token_id,
        )

    for out in outputs:
        gen     = out[inputs["input_ids"].shape[1]:]
        decoded = eval_tokenizer.decode(gen, skip_special_tokens=True).strip()
        if "<think>" in decoded:
            decoded = decoded.split("</think>")[-1].strip()
        hypotheses.append(decoded)

# Save per-sentence results
os.makedirs(EVAL_OUT_DIR, exist_ok=True)
with open(SCORED_JSONL, "w", encoding="utf-8") as f:
    for src, ref, hyp in zip(sources, references, hypotheses):
        f.write(json.dumps({"source": src, "reference": ref, "hypothesis": hyp},
                           ensure_ascii=False) + "\n")
print(f"Hypotheses saved to {SCORED_JSONL}")

del eval_model, base_model
gc.collect()
torch.cuda.empty_cache()

# ── Metrics ───────────────────────────────────────────────────────────────────
print("Computing BLEU...")
bleu_score  = round(sacrebleu.corpus_bleu(hypotheses, [references]).score, 4)
print(f"  BLEU: {bleu_score}")

print("Computing ChrF...")
chrf_score  = round(sacrebleu.corpus_chrf(hypotheses, [references]).score, 4)
print(f"  ChrF: {chrf_score}")

print("Computing COMET-DA (wmt22-comet-da)...")
comet_path  = download_model(COMET_MODEL)
comet_mdl   = load_from_checkpoint(comet_path)
comet_data  = [{"src": s, "mt": h, "ref": r}
               for s, h, r in zip(sources, hypotheses, references)]
comet_score = round(comet_mdl.predict(comet_data, batch_size=16, gpus=1).system_score, 4)
print(f"  COMET-DA: {comet_score}")
del comet_mdl
torch.cuda.empty_cache()

scores = {"bleu": bleu_score, "chrf": chrf_score, "comet_wmt22": comet_score}
with open(SUMMARY_JSON, "w") as f:
    json.dump({"additional_test_qwen_sft": scores}, f, indent=2)
print(f"Summary scores saved to {SUMMARY_JSON}")

# ── Comparison table ──────────────────────────────────────────────────────────
BASELINE_SCORES_FILE = f"{MODEL_PATH}/final_eval_results/emea/test_scores.json"
try:
    with open(BASELINE_SCORES_FILE) as f:
        baseline = json.load(f)
    plain = baseline.get("qwen_sft_plain", {})
    terms = baseline.get("qwen_sft_terms", {})
except FileNotFoundError:
    plain = terms = {}

print("\n" + "=" * 85)
print("  COMPARISON — Qwen3-8B EMEA test set")
print("=" * 85)
print(
    "  CONFIG DIFFERENCES vs baseline:"
    "\n    1. target_modules : q/k/v/o_proj   vs  all-linear"
    "\n    2. Eff. batch size : 4 (2×2)        vs  32 (8×4)"
    "\n    3. LoRA rank       : 64             vs  8"
    "\n    4. LoRA alpha      : 16 (α/r=0.25)  vs  64 (α/r=8.0)"
    "\n    5. LoRA dropout    : 0.1            vs  0.05"
    "\n    6. Learning rate   : 2e-5           vs  4.018e-4"
    "\n    7. Epochs          : 2              vs  3"
    "\n    8. Warmup          : ratio=0.03     vs  steps=50"
    "\n    9. top_p=0.9 passed at generation (no effect — greedy decoding used,"
    "\n       consistent with existing eval pipeline which also uses do_sample=False)"
)
print("=" * 85)
print(f"{'Metric':<12} {'Baseline plain':>16} {'Baseline terms':>16} {'Additional (2ep,r=64)':>22}")
print("-" * 85)
for metric, key in [("BLEU", "bleu"), ("ChrF", "chrf"), ("COMET-DA", "comet_wmt22")]:
    print(
        f"{metric:<12} "
        f"{str(plain.get(key, '—')):>16} "
        f"{str(terms.get(key, '—')):>16} "
        f"{str(scores.get(key, '—')):>22}"
    )
print("=" * 85)
