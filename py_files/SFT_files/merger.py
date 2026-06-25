from peft import PeftModel
from transformers import AutoModelForCausalLM

base_model_dir = "Qwen/Qwen3-8B"
#base_model_dir = "Unbabel/TowerInstruct-7B-v0.2"
#base_model_dir = "mistralai/Ministral-8B-Instruct-2410"
base_model = AutoModelForCausalLM.from_pretrained(base_model_dir)

merged_model = PeftModel.from_pretrained(base_model, "models/SFT_Qwen3_terms/")
#merged_model = PeftModel.from_pretrained(base_model, "models/SFT_TowerInstruct_final/")
#merged_model = PeftModel.from_pretrained(base_model, "models/SFT_Ministral_terms/")

merged_model = merged_model.merge_and_unload()
merged_model.save_pretrained("models/SFT_Qwen3_terms_merged_final")
#merged_model.save_pretrained("models/SFT_TowerInstruct_terms_merged_final")
#merged_model.save_pretrained("models/SFT_Ministral_terms_merged_final")