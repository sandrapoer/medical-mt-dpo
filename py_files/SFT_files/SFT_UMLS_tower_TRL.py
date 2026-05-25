import os
import torch
from dotenv import load_dotenv
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig

load_dotenv()

PROCESSED_PATH = os.getenv("DATA_PROCESSED_DIR").rstrip("/")
MODEL_PATH     = os.getenv("MODELS_DIR").rstrip("/")

BASE_MODEL = "Unbabel/TowerInstruct-7B-v0.2"
OUTPUT_DIR = f"{MODEL_PATH}/SFT_UMLS_TRL_TowerInstruct_terms"


# Tokenizer
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"


# Dataset — terminology-enriched prompts
dataset = load_dataset(
    "json",
    data_files={
        "train": f"{PROCESSED_PATH}/train/messages_train_terms.jsonl",
        "validation": f"{PROCESSED_PATH}/val/messages_val_terms.jsonl",
    }
)


class CompletionOnlyCollator:
    def __init__(self, tokenizer, response_template="<|im_start|>assistant\n"):
        self.tokenizer = tokenizer
        self.response_token_ids = tokenizer.encode(response_template, add_special_tokens=False)
        self.pad_token_id = tokenizer.pad_token_id

    def __call__(self, batch):
        input_ids_list = [torch.tensor(x["input_ids"]) for x in batch]
        max_len = max(t.size(0) for t in input_ids_list)
        input_ids = torch.stack([
            torch.nn.functional.pad(t, (0, max_len - t.size(0)), value=self.pad_token_id)
            for t in input_ids_list
        ])
        attention_mask = (input_ids != self.pad_token_id).long()
        labels = input_ids.clone()
        for i, label_seq in enumerate(labels):
            seq = label_seq.tolist()
            response_start = None
            for j in range(len(seq) - len(self.response_token_ids) + 1):
                if seq[j: j + len(self.response_token_ids)] == self.response_token_ids:
                    response_start = j + len(self.response_token_ids)
                    break
            if response_start is not None:
                labels[i, :response_start] = -100
            else:
                labels[i, :] = -100
        labels[input_ids == self.pad_token_id] = -100
        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


# Model (4-bit QLoRA)
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
)

model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
)
model = prepare_model_for_kbit_training(model)


# LoRA — best params from Optuna TRL Trial 11
lora_config = LoraConfig(
    r=8,
    lora_alpha=64,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

collator = CompletionOnlyCollator(tokenizer)

training_args = SFTConfig(
    output_dir=OUTPUT_DIR,
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=4,
    eval_accumulation_steps=4,
    learning_rate=0.00040183357574217137,
    warmup_steps=50,
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
    dataset_text_field="messages",
    max_length=512,
)

trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["validation"],
    processing_class=tokenizer,
    peft_config=lora_config,
    data_collator=collator,
)

print("Starting training...")
trainer.train()
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print(f"Model and tokenizer saved to {OUTPUT_DIR}")