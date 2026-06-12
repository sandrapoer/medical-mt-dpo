import os
import gc
import argparse
import torch
from dotenv import load_dotenv
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, prepare_model_for_kbit_training
from trl import DPOTrainer, DPOConfig
 
CONFIGS = {
    "ministral": dict(
        merged="SFT_Ministral_terms_merged",
        data="dpo/dpo_train_terms_ministral.jsonl",
        stem="DPO_Ministral_terms",
    ),
    "tower": dict(
        merged="SFT_TowerInstruct_terms_merged",
        data="dpo/dpo_train_terms_tower.jsonl",
        stem="DPO_TowerInstruct_terms",
    ),
    "qwen": dict(
        merged="SFT_Qwen3_terms_merged",
        data="dpo/dpo_train_terms_qwen.jsonl",
        stem="DPO_Qwen3_terms",
    ),
}
 
DEFAULT_BETAS = [0.01, 0.05, 0.1, 0.5]
 
 
def parse_args():
    p = argparse.ArgumentParser(description="Terms (UMLS) DPO trainer.")
    p.add_argument("--model", required=True, choices=["tower", "ministral", "qwen"])
    p.add_argument("--betas", type=float, nargs="+", default=DEFAULT_BETAS)
    return p.parse_args()
 
 
def bnb_config():
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
 
 
def build_peft_config():
    return LoraConfig(
        r=128, lora_alpha=64, target_modules="all-linear",
        lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
    )
 
 
def build_dpo_config(output_dir, beta):
    return DPOConfig(
        output_dir=output_dir,
        beta=beta,
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
 
 
def main():
    args = parse_args()
    load_dotenv()
    processed = os.getenv("DATA_PROCESSED_DIR").rstrip("/")
    models_dir = os.getenv("MODELS_DIR").rstrip("/")
 
    cfg = CONFIGS[args.model]
    merged_path = f"{models_dir}/{cfg['merged']}"
    data_path = f"{processed}/{cfg['data']}"
 
    # check paths before starting training
    if not os.path.isdir(merged_path):
        raise SystemExit(f"Merged model not found: {merged_path}")
    if not os.path.isfile(data_path):
        raise SystemExit(f"DPO data not found: {data_path}")
 
    print(f"model={args.model} (terms)")
    print(f"  merged: {merged_path}")
    print(f"  data:   {data_path}")
    print(f"  stem:   {cfg['stem']}")
 
    tokenizer = AutoTokenizer.from_pretrained(
        merged_path, padding_side="left", trust_remote_code=True
    )
    tokenizer.pad_token = tokenizer.eos_token
 
    raw = load_dataset("json", data_files={"train": data_path}, split="train")
    split = raw.train_test_split(test_size=0.05, seed=42)
    train_ds, eval_ds = split["train"], split["test"]
    print(f"  Train: {len(train_ds)} pairs | Eval: {len(eval_ds)} pairs")
 
    for beta in args.betas:
        output_dir = f"{models_dir}/{cfg['stem']}_beta{beta}"
        print(f"\n{'='*50}\n  DPO beta={beta}\n{'='*50}")
 
        model = AutoModelForCausalLM.from_pretrained(
            merged_path,
            quantization_config=bnb_config(),
            device_map={"": 0},
            trust_remote_code=True,
        )
        model = prepare_model_for_kbit_training(model)
 
        trainer = DPOTrainer(
            model=model,
            ref_model=None,
            args=build_dpo_config(output_dir, beta),
            peft_config=build_peft_config(),
            train_dataset=train_ds,
            eval_dataset=eval_ds,
            processing_class=tokenizer,
        )
        trainer.train()
        trainer.save_model(output_dir)
        tokenizer.save_pretrained(output_dir)
        print(f"  Saved -> {output_dir}")
 
        del trainer, model
        gc.collect()
        torch.cuda.empty_cache()
 
    print("\nAll beta runs completed.")
 
 
if __name__ == "__main__":
    main()
