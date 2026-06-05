import os
import torch
from dotenv import load_dotenv
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, prepare_model_for_kbit_training
from trl import DPOTrainer, DPOConfig

load_dotenv()

PROCESSED_PATH = os.getenv("DATA_PROCESSED_DIR").rstrip("/")
MODEL_PATH = os.getenv("MODELS_DIR").rstrip("/")

MERGED_MODEL = f"{MODEL_PATH}/SFT_Qwen3_merged"

tokenizer = AutoTokenizer.from_pretrained(
    MERGED_MODEL,
    padding_side="left",
    trust_remote_code=True,
)
tokenizer.pad_token = tokenizer.eos_token

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
)

print("Loading merged SFT Qwen3 model...")
model = AutoModelForCausalLM.from_pretrained(
    MERGED_MODEL,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
)
model = prepare_model_for_kbit_training(model)

peft_config = LoraConfig(
    r=128,
    lora_alpha=64,
    target_modules="all-linear",
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

dataset = load_dataset(
    "json",
    data_files={"train": f"{PROCESSED_PATH}/dpo/dpo_train_qwen.jsonl"},
    split="train",
)

split = dataset.train_test_split(test_size=0.05, seed=42)
train_dataset = split["train"]
eval_dataset  = split["test"]

print(f"Train: {len(train_dataset)} pairs | Eval: {len(eval_dataset)} pairs")

BETAS = [0.01, 0.05, 0.1, 0.5]

for BETA in BETAS:
    OUTPUT_DIR = f"{MODEL_PATH}/DPO_Qwen3_umls_beta{BETA}"
    print(f"\n{'='*50}")
    print(f"  Starting DPO training — beta={BETA}")
    print(f"{'='*50}")

    dpo_config = DPOConfig(
        output_dir=OUTPUT_DIR,
        beta=BETA,
        num_train_epochs=1,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=8,
        learning_rate=5e-7,
        lr_scheduler_type="cosine",
        warmup_steps=50,
        bf16=True,
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=200,
        save_strategy="steps",
        save_steps=200,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        max_length=512,
        report_to="none",
        disable_tqdm=False,
    )

    trainer = DPOTrainer(
        model=model,
        ref_model=None,
        args=dpo_config,
        peft_config=peft_config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
    )

    trainer.train()
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"  DPO model saved to {OUTPUT_DIR}")

print("\nAll beta runs completed.")