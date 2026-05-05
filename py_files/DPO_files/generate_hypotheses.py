import os
import json
import torch
from dotenv import load_dotenv
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
from tqdm import tqdm

load_dotenv()

# Start: Testing a few samples
TEST_MODE = False


PROCESSED_PATH = os.getenv("DATA_PROCESSED_DIR").rstrip("/")
MODEL_PATH = os.getenv("MODELS_DIR").rstrip("/")

# Path to your best SFT checkpoint
SFT_MODEL_PATH = f"{MODEL_PATH}/SFT_TowerInstruct_final/checkpoint-1250"
BASE_MODEL_NAME = "Unbabel/TowerInstruct-7B-v0.2"
OUTPUT_FILE = f"{PROCESSED_PATH}/dpo/hypotheses.jsonl"
NUM_HYPOTHESES = 8
TEMPERATURE = 0.7
TOP_P = 0.9
MAX_NEW_TOKENS = 128
BATCH_SIZE = 16 


os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

# Tokenizer loaded from base model 
tokenizer = AutoTokenizer.from_pretrained(
    BASE_MODEL_NAME,
    padding_side="left",  # left padding required for causal LM generation
    trust_remote_code=True,
)
tokenizer.pad_token = tokenizer.eos_token


# base loaded in 4-bit, 
# LoRA applied from SFT checkpoint
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
)

base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_NAME,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
)
model = PeftModel.from_pretrained(base_model, SFT_MODEL_PATH)
model.eval()


dataset = load_dataset(
    "json",
    data_files={"train": f"{PROCESSED_PATH}/train/messages_train.jsonl"},
    split="train",
)

# Testing
if TEST_MODE:
    dataset = dataset.select(range(5))



def build_prompt(example):
    # Reconstruct the ChatML prompt from the messages field
    messages = example["messages"]
    user_msg = next(m for m in messages if m["role"] == "user")
    return f"<|im_start|>user\n{user_msg['content']}<|im_end|>\n<|im_start|>assistant\n"


def get_source(example):
    messages = example["messages"]
    user_msg = next(m for m in messages if m["role"] == "user")
    for line in user_msg["content"].split("\n"):
        if line.startswith("English:"):
            return line[len("English:"):].strip()
    return ""

# Generation
completed = set()
if os.path.exists(OUTPUT_FILE):
    with open(OUTPUT_FILE, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            completed.add(obj["source"])
    print(f"Resuming — {len(completed)} examples already done.")

outfile = open(OUTPUT_FILE, "a", encoding="utf-8")

prompts  = []
sources  = []
indices  = []

for i, example in enumerate(tqdm(dataset, desc="Generating hypotheses")):
    source = get_source(example)
    if source in completed:
        continue
    prompts.append(build_prompt(example))
    sources.append(source)
    indices.append(i)

    # Process in batches
    if len(prompts) == BATCH_SIZE or i == len(dataset) - 1:
        inputs = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        ).to("cuda")

        with torch.no_grad():
            # Generate NUM_HYPOTHESES samples per prompt
            # do_sample=True + temperature enables diverse sampling
            outputs = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=True,
                temperature=TEMPERATURE,
                top_p=TOP_P,
                num_return_sequences=NUM_HYPOTHESES,
                pad_token_id=tokenizer.pad_token_id,
                repetition_penalty=1.3,
                no_repeat_ngram_size=4,
                eos_token_id=tokenizer.eos_token_id,
            )

        # outputs shape: (BATCH_SIZE * NUM_HYPOTHESES, seq_len)
        # reshape to (BATCH_SIZE, NUM_HYPOTHESES, seq_len)
        input_len = inputs["input_ids"].shape[1]
        batch_size = len(prompts)

        for b in range(batch_size):
            hyps = []
            for h in range(NUM_HYPOTHESES):
                idx = b * NUM_HYPOTHESES + h
                # Decode only the newly generated tokens, not the prompt
                generated = outputs[idx][input_len:]
                decoded = tokenizer.decode(generated, skip_special_tokens=True).strip()
                # Truncate at first sentence end
                if "." in decoded:
                    decoded = decoded[:decoded.index(".") + 1].strip()
                hyps.append(decoded)

            record = {
                "source": sources[b],
                "hypotheses": hyps,  # list of 8 translation candidates
            }
            outfile.write(json.dumps(record, ensure_ascii=False) + "\n")

        prompts, sources, indices = [], [], []

outfile.close()
print(f"Done. Hypotheses saved to {OUTPUT_FILE}")