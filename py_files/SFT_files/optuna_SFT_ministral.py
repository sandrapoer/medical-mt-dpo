import os
import gc
import torch
import optuna
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
BASE_MODEL = "mistralai/Ministral-8B-Instruct-2410"
OUTPUT_DIR = f"{MODEL_PATH}/optuna_ministral"
DB_PATH = f"{MODEL_PATH}/optuna_ministral.db"
N_TRIALS = 15
GPU_ID = 0



tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"



raw_dataset = load_dataset(
    "json",
    data_files={
        "train":      f"{PROCESSED_PATH}/train/messages_train.jsonl",
        "validation": f"{PROCESSED_PATH}/val/messages_val.jsonl",
    }
)


def tokenize(example):
    messages = example["messages"]
    user_msg = next(m for m in messages if m["role"] == "user")
    asst_msg = next(m for m in messages if m["role"] == "assistant")
    # Ministral format: <s>[INST]user[/INST]assistant</s>
    # different from Qwen and TowerInstruct -> use <s>user</s>assistant</s>
    text = f"<s>[INST]{user_msg['content']}[/INST]{asst_msg['content']}</s>"
    return tokenizer(text, truncation=True, max_length=512)


dataset = raw_dataset.map(tokenize, remove_columns=raw_dataset["train"].column_names)



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



def objective(trial):
    lr         = trial.suggest_float("learning_rate", 1e-5, 5e-4, log=True)
    warmup     = trial.suggest_int("warmup_steps", 50, 200)
    lora_r     = trial.suggest_categorical("lora_r", [8, 16, 32])
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

    training_args = TrainingArguments(
        output_dir=trial_dir,
        num_train_epochs=1,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=8,
        eval_accumulation_steps=4,
        learning_rate=lr,
        warmup_steps=warmup,
        lr_scheduler_type="cosine",
        bf16=True,
        logging_steps=50,
        eval_strategy="epoch",
        save_strategy="no",
        report_to="none",
        disable_tqdm=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        processing_class=tokenizer,
        data_collator=collator,
    )

    trainer.train()
    eval_results = trainer.evaluate()
    eval_loss = eval_results["eval_loss"]

    del model, base_model, trainer
    gc.collect()
    torch.cuda.empty_cache()

    return eval_loss


# Optuna study
if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=42),
        study_name="ministral_sft_optuna",
        storage=f"sqlite:///{DB_PATH}",
        load_if_exists=True,
    )

    study.optimize(objective, n_trials=N_TRIALS)

    print("\nBest trial:")
    print(f"  eval_loss : {study.best_trial.value:.4f}")
    print("  Params:")
    for k, v in study.best_trial.params.items():
        print(f"    {k}: {v}")