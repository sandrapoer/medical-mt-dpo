import os
import gc
import torch
import optuna
from dotenv import load_dotenv
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig

load_dotenv()

PROCESSED_PATH = os.getenv("DATA_PROCESSED_DIR").rstrip("/")
MODEL_PATH = os.getenv("MODELS_DIR").rstrip("/")
BASE_MODEL = "Qwen/Qwen3-8B"
OUTPUT_DIR = f"{MODEL_PATH}/optuna_qwen3"
DB_PATH = f"{MODEL_PATH}/optuna_qwen3.db"
N_TRIALS = 15
GPU_ID = 0 # running on one GPU, possible on multiple


tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"


raw_dataset = load_dataset(
    "json",
    data_files={
        "train": f"{PROCESSED_PATH}/train/messages_train.jsonl",
        "validation": f"{PROCESSED_PATH}/val/messages_val.jsonl",
    }
)


def format_sample(sample):
    text = tokenizer.apply_chat_template(
        sample["messages"],
        tokenize=False,
        add_generation_prompt=False,
        enable_thinking=False,
    )
    return {"text": text}


dataset = raw_dataset.map(format_sample, remove_columns=raw_dataset["train"].column_names)



class CompletionOnlyCollator:
    def __init__(self, tokenizer, max_length=512):
        self.tokenizer  = tokenizer
        self.max_length = max_length
        self.response_template = "<|im_start|>assistant\n"

    def __call__(self, features):
        texts  = [f["text"] for f in features]
        batch  = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length,
        )
        labels = batch["input_ids"].clone()

    
        template_ids = self.tokenizer.encode(
            self.response_template, add_special_tokens=False
        )
        tlen = len(template_ids)

        for i, label_row in enumerate(labels):
            ids = label_row.tolist()
            mask_until = 0
            for j in range(len(ids) - tlen + 1):
                if ids[j : j + tlen] == template_ids:
                    mask_until = j + tlen
            labels[i, :mask_until] = -100

        # Also mask padding
        labels[batch["attention_mask"] == 0] = -100

        return {
            "input_ids":      batch["input_ids"],
            "attention_mask": batch["attention_mask"],
            "labels":         labels,
        }


# OPTUNA objective
def objective(trial):
    lr = trial.suggest_float("learning_rate", 1e-5, 5e-4, log=True)
    warmup = trial.suggest_int("warmup_steps", 50, 200)
    lora_r = trial.suggest_categorical("lora_r", [8, 16, 32])
    lora_alpha = trial.suggest_categorical("lora_alpha", [16, 32, 64])

    trial_dir = f"{OUTPUT_DIR}/trial_{trial.number}"
    os.makedirs(trial_dir, exist_ok=True)

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=bnb_config,
        device_map={"": GPU_ID},
        trust_remote_code=True,
    )
    base_model = prepare_model_for_kbit_training(base_model)

    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules="all-linear",
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(base_model, lora_config)

    collator = CompletionOnlyCollator(tokenizer)

    training_args = SFTConfig(
        output_dir=trial_dir,
        num_train_epochs=1,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=8,
        learning_rate=lr,
        warmup_steps=warmup,
        lr_scheduler_type="cosine",
        bf16=True,
        logging_steps=50,
        eval_strategy="epoch",
        save_strategy="no",
        report_to="none",
        disable_tqdm=False,
        dataset_text_field="text",
        max_length=512,
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        data_collator=collator,
    )

    trainer.train()
    eval_results = trainer.evaluate()
    eval_loss = eval_results["eval_loss"]

    del model, base_model, trainer
    gc.collect()
    torch.cuda.empty_cache()

    return eval_loss



# Study
if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=42),
        study_name="qwen3_sft_optuna",
        storage=f"sqlite:///{DB_PATH}",
        load_if_exists=True,
    )

    study.optimize(objective, n_trials=N_TRIALS)

    print("\nBest trial:")
    print(f"  eval_loss : {study.best_trial.value:.4f}")
    print("  Params:")
    for k, v in study.best_trial.params.items():
        print(f"    {k}: {v}")