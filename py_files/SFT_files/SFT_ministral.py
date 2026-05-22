import os
import torch
from dotenv import load_dotenv
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

load_dotenv()

PROCESSED_PATH = os.getenv("DATA_PROCESSED_DIR").rstrip("/")
MODEL_PATH = os.getenv("MODELS_DIR").rstrip("/")

MODEL_NAME = "mistralai/Ministral-8B-Instruct-2410"
OUTPUT_DIR = f"{MODEL_PATH}/SFT_Ministral_final"


tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, padding_side="left", trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token


bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map={"": 0},
    trust_remote_code=True,
)
model = prepare_model_for_kbit_training(model)

# wiht optuna best hyperparameters
lora_config = LoraConfig(
    r=8,
    lora_alpha=64,
    target_modules="all-linear",
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()


dataset = load_dataset(
    "json",
    data_files={
        "train": f"{PROCESSED_PATH}/train/messages_train.jsonl",
        "validation": f"{PROCESSED_PATH}/val/messages_val.jsonl",
    }
)


def tokenize(example):
    messages = example["messages"]
    user_msg = next(m for m in messages if m["role"] == "user")
    asst_msg = next(m for m in messages if m["role"] == "assistant")
    # Ministral format: <s>[INST]user[/INST]assistant</s>
    text = f"<s>[INST]{user_msg['content']}[/INST]{asst_msg['content']}</s>"
    return tokenizer(text, truncation=True, max_length=512)


tokenized = dataset.map(tokenize, remove_columns=dataset["train"].column_names)


class CompletionOnlyCollator:
    def __init__(self, tokenizer, response_template="[/INST]"):
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
            "input_ids":      input_ids,
            "attention_mask": attention_mask,
            "labels":         labels,
        }


collator = CompletionOnlyCollator(tokenizer)

# with optuna best hyperparameters
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=4,
    eval_accumulation_steps=4,
    learning_rate=9.418986507624857e-05,
    warmup_steps=60,
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