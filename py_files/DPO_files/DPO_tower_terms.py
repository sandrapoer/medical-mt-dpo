import os
import torch
import re
from dotenv import load_dotenv
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig
from trl import DPOTrainer, DPOConfig

load_dotenv()

print(torch.cuda.is_available())

PROCESSED_PATH = os.getenv("DATA_PROCESSED_DIR").rstrip("/")
MODEL_PATH     = os.getenv("MODELS_DIR").rstrip("/")

MERGED_MODEL = f"{MODEL_PATH}/SFT_TowerInstruct_terms_merged"

tokenizer = AutoTokenizer.from_pretrained("Unbabel/TowerInstruct-7B-v0.2", padding_side="left", trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16)

print("Loading merged SFT TowerInstruct terms model...")
model = AutoModelForCausalLM.from_pretrained(MERGED_MODEL, quantization_config=bnb_config,
    device_map={"": 0}, trust_remote_code=True)

peft_config = LoraConfig(r=128, lora_alpha=64, target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.05, bias="none", task_type="CAUSAL_LM")

dataset = load_dataset("json",
    data_files={"train": f"{PROCESSED_PATH}/dpo/dpo_train_terms_tower.jsonl"})

dataset1 = []
for sample in dataset["train"]:
    prompt = sample["prompt"]
    cleaned = re.sub(r"<\|im_start\|>user\n?", "", prompt)
    cleaned = re.sub(r"<\|im_start\|>assistant\n?", "", cleaned)
    cleaned = re.sub(r"<\|im_end\|>\n?", "", cleaned)
    dataset1.append({
        "prompt":   [{"role": "user",      "content": cleaned}],
        "chosen":   [{"role": "assistant", "content": sample["chosen"]}],
        "rejected": [{"role": "assistant", "content": sample["rejected"]}],
    })

dpo_dataset = Dataset.from_list(dataset1)

OUTPUT_DIR = f"{MODEL_PATH}/DPO_TowerInstruct_restructured_beta_0.01_3epochs"

dpo_config = DPOConfig(output_dir=OUTPUT_DIR, beta=0.01, num_train_epochs=3,
    per_device_train_batch_size=2, per_device_eval_batch_size=2,
    gradient_accumulation_steps=8, gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False},
    learning_rate=5e-6, lr_scheduler_type="cosine",
    warmup_steps=50, bf16=True, logging_steps=10,
    save_total_limit=2, max_length=512, report_to="none", disable_tqdm=False)

trainer = DPOTrainer(model=model, ref_model=None, args=dpo_config,
    peft_config=peft_config, train_dataset=dpo_dataset,
    processing_class=tokenizer)

trainer.train()
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print(f"  Saved to {OUTPUT_DIR}")