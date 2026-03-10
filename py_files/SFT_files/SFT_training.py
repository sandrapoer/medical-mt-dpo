from dotenv import load_dotenv
import os
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig
import bitsandbytes as bnb

load_dotenv()
processed_path = os.getenv("DATA_PROCESSED_DIR")
model_path = os.getenv("MODELS_DIR")

MODEL_NAME = "Unbabel/TowerInstruct-v0.2"
OUTPUT_DIR = os.path.join(model_path, "SFT_TowerInstruct_v0.2")

# Load Tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# Load Model in 4-bit QLoRA
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True,
    ),
model = prepare_model_for_kbit_training(model)

# LoRA Configuration
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# Load Formatted Dataset
dataset = load_dataset(
    "json",
    data_files={
        "train": f"{processed_path}/train/formatted_train.jsonl",
        "validation": f"{processed_path}/val/formatted_val.jsonl",
    }
)

def format_sample(sample):
    return sample["prompt"] + sample["completion"] + tokenizer.eos_token

dataset = dataset.map(format_sample)

# Only compute loss on the completion part
response_template = "\nFrench:\n"
collator = DataCollatorForCompletionOnlyLM(
    response_template=response_template,
    tokenizer=tokenizer,
)

training_args = SFTConfig(
    disable_tqdm=False,
    output_dir=OUTPUT_DIR,
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    lr_scheduler_type="cosine",
    warmup_ratio=0.03,
    bf16=True,
    logging_steps=10,
    evaluation_strategy="steps",
    eval_steps=500,
    save_strategy="steps",
    save_steps=500,
    save_total_limit=2,
    load_best_model_at_end=True,
    report_to="none",
    max_seq_length=512,
)

trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["validation"],
)

print("Starting training...")
trainer.train()
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print(f"Model and tokenizer saved to {OUTPUT_DIR}")