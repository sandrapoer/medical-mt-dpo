import json, os, re
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()
DATA_PROCESSED_DIR = Path(os.getenv("DATA_PROCESSED_DIR"))

IN_PATH  = DATA_PROCESSED_DIR / "dpo/umls_dpo_pairs_qwen.jsonl"
OUT_PATH = DATA_PROCESSED_DIR / "dpo/dpo_train_terms_qwen.jsonl"

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

        # strip <think> tags from Qwen hypotheses
        import re
        chosen   = re.sub(r"<think>.*?</think>\n?", "", chosen,   flags=re.DOTALL).strip()
        rejected = re.sub(r"<think>.*?</think>\n?", "", rejected, flags=re.DOTALL).strip()

        prompt = (
            f"<|im_start|>user\n{instruction}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )

        fout.write(json.dumps({
            "prompt": prompt, "chosen": chosen, "rejected": rejected,
        }, ensure_ascii=False) + "\n")
        written += 1

print(f"Written: {written} pairs to {OUT_PATH}")
