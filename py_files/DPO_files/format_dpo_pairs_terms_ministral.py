import json, os, re
from pathlib import Path
from dotenv import load_dotenv
from transformers import AutoTokenizer

load_dotenv()
DATA_PROCESSED_DIR = Path(os.getenv("DATA_PROCESSED_DIR"))

IN_PATH  = DATA_PROCESSED_DIR / "dpo/umls_dpo_pairs_ministral.jsonl"
OUT_PATH = DATA_PROCESSED_DIR / "dpo/dpo_train_terms_ministral.jsonl"

tok = AutoTokenizer.from_pretrained("mistralai/Ministral-8B-Instruct-2410", trust_remote_code=True)

PLAIN_INSTRUCTION = (
    "Translate the English source text to French. "
    "Rules: output strictly the French translation. "
    "Do NOT repeat the English text. "
    "Do NOT include labels like \"English\" or \"French\". "
    "Do NOT add quotation marks, explanations or any extra text. "
    "Now translate the following:\nEnglish: {source}\nFrench:\n"
)

TERM_INSTRUCTION = (
    "Glossaries:\n{glossary_lines}\n\n"
    "Translate the English source text to French following the provided translation glossaries. "
    "Rules: output strictly the French translation. "
    "Do NOT repeat the English text. "
    "Do NOT include labels like \"English\" or \"French\". "
    "Do NOT add quotation marks, explanations or any extra text. "
    "Now translate the following:\nEnglish: {source}\nFrench:\n"
)

written = 0
with open(IN_PATH) as fin, open(OUT_PATH, "w") as fout:
    for line in fin:
        rec = json.loads(line)
        source   = rec["source"]
        chosen   = rec["chosen"]
        rejected = rec["rejected"]
        entities = rec.get("matched_entities", [])
        fr_syns  = rec.get("fr_synonyms", [])

        term_pairs = [(en, syns[0]) for en, syns in zip(entities, fr_syns) if syns]

        if term_pairs:
            glossary_lines = "\n".join(f'"{en}" -> "{fr}"' for en, fr in term_pairs)
            instruction = TERM_INSTRUCTION.format(glossary_lines=glossary_lines, source=source)
        else:
            instruction = PLAIN_INSTRUCTION.format(source=source)

        prompt = tok.apply_chat_template(
            [{"role": "user", "content": instruction}],
            tokenize=False,
            add_generation_prompt=True,
        )

        fout.write(json.dumps({
            "prompt": prompt, "chosen": chosen, "rejected": rejected,
        }, ensure_ascii=False) + "\n")
        written += 1

print(f"Written: {written} pairs to {OUT_PATH}")
