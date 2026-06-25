from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
'''
#base_model_dir = "Qwen/Qwen3-8B"
#base_model_dir = "Unbabel/TowerInstruct-7B-v0.2"
#base_model_dir = "mistralai/Ministral-8B-Instruct-2410"

tokenizer = AutoTokenizer.from_pretrained(base_model_dir)
base_model = AutoModelForCausalLM.from_pretrained(base_model_dir)

merged_model = PeftModel.from_pretrained(base_model, "models/SFT_Qwen3_new_run/")
#merged_model = PeftModel.from_pretrained(base_model, "models/SFT_TowerInstruct_final/")
#merged_model = PeftModel.from_pretrained(base_model, "models/SFT_Ministral_terms/")

merged_model = merged_model.merge_and_unload()
merged_model.save_pretrained("models/SFT_Qwen3_new_run_merged_final")
#merged_model.save_pretrained("models/SFT_TowerInstruct_terms_merged_final")
#merged_model.save_pretrained("models/SFT_Ministral_terms_merged_final")

tokenizer.save_pretrained("models/SFT_Qwen3_new_run_merged_final")

# --- DPO Tower ---
base_model_dir = "models/SFT_TowerInstruct_terms_merged"
tokenizer = AutoTokenizer.from_pretrained("Unbabel/TowerInstruct-7B-v0.2", trust_remote_code=True)
base_model = AutoModelForCausalLM.from_pretrained(base_model_dir, torch_dtype=torch.bfloat16)
merged_model = PeftModel.from_pretrained(base_model, "models/DPO_TowerInstruct_beta0.01")
merged_model = merged_model.merge_and_unload()
merged_model.save_pretrained("models/DPO_TowerInstruct_beta0.01_merged")
tokenizer.save_pretrained("models/DPO_TowerInstruct_beta0.01_merged")

# --- DPO Qwen ---
base_model_dir = "models/SFT_Qwen3_terms_merged_final"
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")
base_model = AutoModelForCausalLM.from_pretrained(base_model_dir, dtype=torch.bfloat16)
merged_model = PeftModel.from_pretrained(base_model, "models/DPO_Qwen3_restructured_beta_0.01_3epochs")
merged_model = merged_model.merge_and_unload()
merged_model.save_pretrained("models/DPO_Qwen3_restructured_beta_0.01_3epochs_merged")
tokenizer.save_pretrained("models/DPO_Qwen3_restructured_beta_0.01_3epochs_merged")
'''

# --- DPO Ministral ---
base_model_dir = "models/SFT_Ministral_terms_merged"
tokenizer = AutoTokenizer.from_pretrained("mistralai/Ministral-8B-Instruct-2410")
base_model = AutoModelForCausalLM.from_pretrained(base_model_dir, dtype=torch.bfloat16)
merged_model = PeftModel.from_pretrained(base_model, "models/DPO_Ministral_beta0.01")
merged_model = merged_model.merge_and_unload()
merged_model.save_pretrained("models/DPO_Ministral_beta0.01_merged")
tokenizer.save_pretrained("models/DPO_Ministral_beta0.01_merged")