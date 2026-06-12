import os
from dotenv import load_dotenv
from transformers import AutoTokenizer
 
load_dotenv()
MODEL_NAME = "mistralai/Ministral-8B-Instruct-2410"
 
tok = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
 
user = "English: The patient was given 5 mg of the drug."
asst = "Le patient a reçu 5 mg du médicament."
text = f"<s>[INST]{user}[/INST]{asst}</s>"
 
ids_default = tok(text)["input_ids"]                       # what training actually fed
ids_no_special = tok(text, add_special_tokens=False)["input_ids"]
 
print(f"BOS id = {tok.bos_token_id} ({tok.bos_token!r})")
print(f"EOS id = {tok.eos_token_id} ({tok.eos_token!r})")
print(f"\nfirst 6 ids WITH default add_special_tokens : {ids_default[:6]}")
print(f"first 6 ids WITHOUT special tokens         : {ids_no_special[:6]}")
 
# SUSPECT 1: double BOS?
dbl = ids_default[:2] == [tok.bos_token_id, tok.bos_token_id] or (
    ids_default[0] == tok.bos_token_id and ids_no_special[0] == tok.bos_token_id
)
print(f"\n[SUSPECT 1] double-BOS likely: {dbl}")
 
# SUSPECT 2: does the mask template actually match inside the full text?
resp_ids = tok.encode("[/INST]", add_special_tokens=False)
def contains(hay, needle):
    n = len(needle)
    return any(hay[i:i+n] == needle for i in range(len(hay)-n+1))
found_default = contains(ids_default, resp_ids)
found_no_special = contains(ids_no_special, resp_ids)
print(f"\nencode('[/INST]') -> {resp_ids}")
print(f"[SUSPECT 2] '[/INST]' subsequence found in training ids: {found_default}")
print(f"            (without special tokens)                  : {found_no_special}")
print("\n>>> If SUSPECT 2 is False, the collator masked EVERY token "
      "(labels all -100) and the model never trained. That is the collapse.")
 