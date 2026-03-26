from dotenv import load_dotenv
import os
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig

load_dotenv()
processed_path = os.getenv("DATA_PROCESSED_DIR").rstrip("/")
model_path = os.getenv("MODELS_DIR")

MODEL_NAME = "Unbabel/TowerInstruct-7B-v0.2"
OUTPUT_DIR = os.path.join(model_path, "SFT_TowerInstruct_v0.2")

# Load Tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
)

# Load Model in 4-bit QLoRA
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
)
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

# Load Dataset
dataset = load_dataset(
    "json",
    data_files={
        "train": f"{processed_path}/train/messages_train.jsonl",
        "validation": f"{processed_path}/val/messages_val.jsonl",
    }
)

class CompletionOnlyCollator:
    """Mask prompt tokens so loss is only computed on the French translation."""
    def __init__(self, tokenizer, response_template="<|im_start|>assistant\n"):
        self.tokenizer = tokenizer
        self.response_token_ids = tokenizer.encode(response_template, add_special_tokens=False)
        self.pad_token_id = tokenizer.pad_token_id

    def __call__(self, batch):
        # Pad sequences to the same length
        input_ids_list = [torch.tensor(x["input_ids"]) for x in batch]
        max_len = max(t.size(0) for t in input_ids_list)

        input_ids = torch.stack([
            torch.nn.functional.pad(t, (0, max_len - t.size(0)), value=self.pad_token_id)
            for t in input_ids_list
        ])

        # Build attention mask: 1 for real tokens, 0 for padding
        attention_mask = (input_ids != self.pad_token_id).long()

        labels = input_ids.clone()

        for i, label_seq in enumerate(labels):
            seq = label_seq.tolist()
            response_start = None
            for j in range(len(seq) - len(self.response_token_ids) + 1):
                if seq[j:j + len(self.response_token_ids)] == self.response_token_ids:
                    response_start = j + len(self.response_token_ids)
                    break
            if response_start is not None:
                labels[i, :response_start] = -100
            else:
                labels[i, :] = -100

        # Mask padding tokens in labels
        labels[input_ids == self.pad_token_id] = -100

        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


collator = CompletionOnlyCollator(tokenizer)

# Training Configuration
training_args = SFTConfig(
    disable_tqdm=False,
    output_dir=OUTPUT_DIR,
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    lr_scheduler_type="cosine",
    warmup_steps=100,
    bf16=True,
    logging_steps=10,
    eval_strategy="steps",
    eval_steps=500,
    save_strategy="steps",
    save_steps=500,
    save_total_limit=2,
    load_best_model_at_end=True,
    report_to="none",
    dataset_text_field="messages",
    max_length=512,
)


trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["validation"],
    processing_class=tokenizer,
    data_collator=collator,
)

print("Starting training...")
trainer.train()
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print(f"Model and tokenizer saved to {OUTPUT_DIR}")