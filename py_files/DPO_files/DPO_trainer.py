import os
import torch
from dotenv import load_dotenv
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, PeftModel
from trl import DPOTrainer, DPOConfig

load_dotenv()

PROCESSED_PATH = os.getenv("DATA_PROCESSED_DIR").rstrip("/")
MODEL_PATH = os.getenv("MODELS_DIR").rstrip("/")

BASE_MODEL_NAME = "Unbabel/TowerInstruct-7B-v0.2"
SFT_CHECKPOINT = f"{MODEL_PATH}/SFT_TowerInstruct_final/checkpoint-1250"
OUTPUT_DIR  = f"{MODEL_PATH}/DPO_TowerInstruct_beta0.01"

BETA = 0.01  # tuning later with 0.01, 0.05, 0.1, 0.5 ...


tokenizer = AutoTokenizer.from_pretrained(
    BASE_MODEL_NAME,
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

base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_NAME,
    quantization_config=bnb_config,
    device_map={"": 0},
    trust_remote_code=True,
)
base_model = prepare_model_for_kbit_training(base_model)


model = PeftModel.from_pretrained(base_model, SFT_CHECKPOINT, is_trainable=True)


# explicit reference model requested by TRL when using PEFT 
ref_base = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_NAME,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
)
ref_model = PeftModel.from_pretrained(ref_base, SFT_CHECKPOINT, is_trainable=False)
ref_model.eval()


dataset = load_dataset(
    "json",
    data_files={"train": f"{PROCESSED_PATH}/dpo/dpo_train_bw.jsonl"},
    split="train",
)

# test here is not tico but test split form dpo_train_bw.jsonl
split = dataset.train_test_split(test_size=0.05, seed=42)
train_dataset = split["train"]
eval_dataset  = split["test"]

print(f"Train: {len(train_dataset)} pairs | Eval: {len(eval_dataset)} pairs")


dpo_config = DPOConfig(
    output_dir=OUTPUT_DIR,
    beta=BETA,
    num_train_epochs=1, # same as mbr paper
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    gradient_accumulation_steps=8,
    learning_rate=5e-7, # same as mbr paper
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
    ref_model=ref_model,
    args=dpo_config,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    processing_class=tokenizer,
)

print("Starting DPO training...")
trainer.train()
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print(f"DPO model saved to {OUTPUT_DIR}")