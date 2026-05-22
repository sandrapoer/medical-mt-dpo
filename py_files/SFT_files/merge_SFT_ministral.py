import os
import torch
from dotenv import load_dotenv
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

load_dotenv()

MODEL_PATH = os.getenv("MODELS_DIR").rstrip("/")
BASE_MODEL_NAME = "mistralai/Ministral-8B-Instruct-2410"
SFT_CHECKPOINT = f"{MODEL_PATH}/SFT_Ministral_final/checkpoint-1875"
MERGED_OUTPUT = f"{MODEL_PATH}/SFT_Ministral_merged"

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
)

print("Loading base model...")
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_NAME,
    quantization_config=bnb_config,
    device_map={"": 0},
    trust_remote_code=True,
)

print("Loading SFT LoRA adapter...")
model = PeftModel.from_pretrained(base_model, SFT_CHECKPOINT)

print("Merging adapter into base model...")
model = model.merge_and_unload()

print(f"Saving merged model to {MERGED_OUTPUT}...")
model.save_pretrained(MERGED_OUTPUT)
tokenizer.save_pretrained(MERGED_OUTPUT)
print("Done.")