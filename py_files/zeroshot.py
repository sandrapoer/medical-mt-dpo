from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig #LlamaForCausalLM
from datasets import load_dataset
from transformers.pipelines.pt_utils import KeyDataset
import sys
import evaluate
import numpy as np
from datasets import Dataset
import codecs
import json
import re 
import torch
import csv
from transformers import pipeline

from itertools import islice

def main(src_id, trg_id, access_token, model_id, model_name, cuda, json_file, mt_output_file):
    max_length = 128
    size = 15
    n_samples = 1
    temperature = 0.6 #0.3
    top_p = 0.9

      

    bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )

    #model = AutoModelForCausalLM.from_pretrained(model_id, 
    #                                             quantization_config=bnb_config,
    #                                             use_cache=False,
    #                                             token=access_token, 
    #                                             device_map="auto") #{"":0}
    tokenizer = AutoTokenizer.from_pretrained(model_id,
                                              #padding_side='left', #TODO only for qwen
                                              token=access_token)

    data_files = {}
    data_files["test"] = json_file
    data = load_dataset("json", data_files=data_files)
    #data = readlines(data_file)

     
    mt_output = codecs.open(mt_output_file, 'w', encoding='utf-8')

    
    
   
    i = 0

    #hyp_txt = [f"hypothesis_{i}" for i in range(n_samples)]

    #head.extend(hyp_txt)

    pipe = pipeline("text-generation", 
                    model=model_id, 
                    model_kwargs={"quantization_config":bnb_config},
                    tokenizer=tokenizer,
                    device_map=cuda, #TODO auto for Qwen
                    batch_size=size)
    
    if model_name == "Ministral-8B-Instruct-2410":
        pipe.tokenizer.pad_token_id = pipe.tokenizer.eos_token_id #for mistall!
    
    
    encoded_dataset = []
    tmp_data = []
    print(model_id)
    for sample in data['test']:
        encoded_dataset.append([sample["messages"][0]]) 

    #print(len(encoded_dataset))
    if model_name == "Qwen3-8B":
        pipe.tokenizer.padding_side='left'
        prompt = pipe.tokenizer.apply_chat_template(encoded_dataset,
                                                    tokenize=False, 
                                                    add_generation_prompt=True,
                                                    enable_thinking=False #TODO only for qwen3!!
                                                    )
    else:
        prompt = pipe.tokenizer.apply_chat_template(encoded_dataset, 
                                                    tokenize=False, 
                                                    add_generation_prompt=True,
                                                    )
    print("tok")
    #print(len(prompt))
    outputs = pipe(prompt, 
                    max_new_tokens=max_length,
                    do_sample=True,
                    num_beams=1,
                    #temperature=temperature,
                    top_p=top_p,
                    top_k=0,
                    num_return_sequences=n_samples,
                    return_full_text=False
                    )

    print('mt')
    print(len(outputs))

    for output in (outputs):
        mt = output[0] ["generated_text"]
        mt_clean = mt.replace("\n", " ").strip()
        print(mt_clean, file=mt_output)
    #output = [prompt, src, trg] #"\n".join(hypotheses)
    #output.extend(hypotheses)
    #print(output)


    return

if __name__ == "__main__":
    if len(sys.argv) != 8:
        print(f'usage: python {sys.argv[0]} <src-id> <trg-id> <model-id> <model-name> <cuda-id> <json-in> <mt-output>')
    else:
        src_id = sys.argv[1] #"German"#"German" 
        trg_id = sys.argv[2] #"English" #"English" 
        access_token = "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" #my token
        model_id = sys.argv[3] #"mistralai/Ministral-8B-Instruct-2410" 
        model_name = sys.argv[4] #"Ministral-8B-Instruct-2410" 
        cuda = {"":int(sys.argv[5])} #{"":0}
        #"Unbabel/TowerInstruct-7B"
        #"mistralai/Ministral-8B-Instruct-2410"
        #"Qwen/Qwen3-8B"        
        json_file = sys.argv[6]
        mt_output_file = sys.argv[7]

        main(src_id, trg_id, access_token, model_id, model_name, cuda, json_file, mt_output_file)