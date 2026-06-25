import json
import torch
import sacrebleu
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, pipeline
from peft import PeftModel
from comet import download_model, load_from_checkpoint

BASE_MODEL = "models/SFT_Qwen3_terms_merged_final" 
DPO_ADAPTER = "models/DPO_Qwen3_restructured_beta_0.01"
TOKENIZER = "Qwen/Qwen3-8B" 
TEST_FILE = "data/processed/test_emea/messages_test_terms.jsonl"
OUT_FILE = "models/DPO_Qwen3_restructured_beta0.01_test_scored.jsonl"
IS_QWEN = True
BATCH_SIZE = 15
MAX_NEW_TOKENS = 256
TOP_P = 0.9


def load_test(path):
    sources, references, msgs = [], [], []
    with open(path, encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            user = next(m for m in obj["messages"] if m["role"] == "user")
            asst = next(m for m in obj["messages"] if m["role"] == "assistant")
            src = ""
            for cl in user["content"].split("\n"):
                if cl.startswith("English:"):
                    src = cl[len("English:"):].strip()
                    break
            sources.append(src)
            references.append(asst["content"].strip())
            msgs.append([user]) 
    return sources, references, msgs


def main():
    sources, references, msgs = load_test(TEST_FILE)
    print(f"{len(msgs)} test examples")

    bnb = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16,
    )

    # load base + DPO adapter, then merge so it behaves like a plain model in the pipeline
    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, quantization_config=bnb, device_map={"": 0},
    )
    model = PeftModel.from_pretrained(base, DPO_ADAPTER)
    model = model.merge_and_unload()
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER, padding_side="left")
    tokenizer.pad_token = tokenizer.eos_token

    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        device_map={"": 0},
        batch_size=BATCH_SIZE,
    )

    prompts = pipe.tokenizer.apply_chat_template(
        msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False,
    )

    outputs = pipe(
        prompts,
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=False,
        num_beams=1,
        top_p=TOP_P,
        num_return_sequences=1,
        return_full_text=False,
    )

    hyps = [o[0]["generated_text"].replace("\n", " ").strip() for o in outputs]

    del model, base
    torch.cuda.empty_cache()

    bleu = round(sacrebleu.corpus_bleu(hyps, [references]).score, 4)
    chrf = round(sacrebleu.corpus_chrf(hyps, [references]).score, 4)

    comet = load_from_checkpoint(download_model("Unbabel/wmt22-comet-da"))
    data = [{"src": s, "mt": h, "ref": r} for s, h, r in zip(sources, hyps, references)]
    cout = comet.predict(data, batch_size=16, gpus=1)
    comet_da = round(cout.system_score, 4)

    with open(OUT_FILE, "w", encoding="utf-8") as f:
        for ln, (s, r, h, sc) in enumerate(zip(sources, references, hyps, cout.scores), 1):
            f.write(json.dumps({"line": ln, "source": s, "reference": r,
                                "hypothesis": h, "seg_comet": round(float(sc), 4)},
                               ensure_ascii=False) + "\n")

    print(f"\nBLEU={bleu}  ChrF={chrf}  COMET-DA={comet_da}")
    print(f"per-segment written to {OUT_FILE}")


if __name__ == "__main__":
    main()