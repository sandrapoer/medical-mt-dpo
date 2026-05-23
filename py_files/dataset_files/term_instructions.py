import json
import os
import time
import spacy
import requests
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

DATA_PROCESSED_DIR = Path(os.getenv("DATA_PROCESSED_DIR"))
UMLS_API_KEY = os.getenv("UMLS_API_KEY")
UMLS_VERSION = "2026AA"
UMLS_BASE = "https://uts-ws.nlm.nih.gov/rest"
REQUEST_DELAY = 0.1

CACHE_PATH = DATA_PROCESSED_DIR / "dpo/umls_cache.json"

SPLITS = [
    {
        "en":      DATA_PROCESSED_DIR / "train/train.en",
        "msgs":    DATA_PROCESSED_DIR / "train/messages_train.jsonl",
        "output":  DATA_PROCESSED_DIR / "train/messages_train_terms.jsonl",
    },
    {
        "en":      DATA_PROCESSED_DIR / "val/val.en",
        "msgs":    DATA_PROCESSED_DIR / "val/messages_val.jsonl",
        "output":  DATA_PROCESSED_DIR / "val/messages_val_terms.jsonl",
    },
]

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


def load_cache(path: Path) -> dict:
    if path.exists():
        with open(path, encoding="utf-8") as f:
            cache = json.load(f)
        print(f"Cache loaded: {len(cache)} entries")
        return cache
    print("No cache found — starting fresh")
    return {}


def save_cache(cache: dict, path: Path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False)


def get_cui(term: str, api_key: str) -> str | None:
    url = f"{UMLS_BASE}/search/{UMLS_VERSION}"
    params = {
        "string": term,
        "apiKey": api_key,
        "returnIdType": "concept",
        "pageSize": 1,
    }
    try:
        r = requests.get(url, params=params, timeout=10)
        r.raise_for_status()
        results = r.json().get("result", {}).get("results", [])
        if results and results[0].get("ui") != "NONE":
            return results[0]["ui"]
    except Exception as e:
        print(f"  CUI lookup failed for '{term}': {e}")
    return None

def get_french_terms(cui: str, api_key: str) -> list:
    url = f"{UMLS_BASE}/content/{UMLS_VERSION}/CUI/{cui}/atoms"
    fr_terms = []
    for page in range(1, 6):
        params = {"apiKey": api_key, "pageSize": 50, "pageNumber": page}
        try:
            r = requests.get(url, params=params, timeout=10)
            if r.status_code == 404:
                break  # no more pages
            r.raise_for_status()
            atoms = r.json().get("result", [])
            if not atoms:
                break
            for atom in atoms:
                if atom.get("language") == "FRE":
                    name = atom.get("name", "").strip().lower()
                    if name:
                        fr_terms.append(name)
            if fr_terms:
                break  # stop after first page with French terms
        except Exception as e:
            print(f"  French atom lookup failed for CUI '{cui}': {e}")
            break
    return list(set(fr_terms))


def lookup_term(term: str, cache: dict, api_key: str) -> list:
    term_lower = term.lower()
    if term_lower in cache:
        return cache[term_lower]
    cui = get_cui(term_lower, api_key)
    time.sleep(REQUEST_DELAY)
    if cui is None:
        cache[term_lower] = []
        return []
    fr_terms = get_french_terms(cui, api_key)
    time.sleep(REQUEST_DELAY)
    cache[term_lower] = fr_terms
    return fr_terms


def load_ner():
    for model in ["en_ner_bc5cdr_md", "en_core_sci_md", "en_core_sci_sm"]:
        try:
            nlp = spacy.load(model)
            print(f"Loaded spaCy model: {model}")
            return nlp
        except OSError:
            continue
    raise RuntimeError("No scispaCy model found.")


def build_prompt(source: str, nlp, cache: dict, api_key: str) -> tuple:
    """
    Returns (prompt_str, n_terms_injected).
    Uses TERM_INSTRUCTION if terms found, PLAIN_INSTRUCTION otherwise.
    Only injects the first (preferred) French synonym per entity.
    """
    doc = nlp(source)
    term_pairs = []
    seen = set()

    for ent in doc.ents:
        en_term = ent.text.lower()
        if en_term in seen:
            continue
        seen.add(en_term)
        fr_syns = lookup_term(en_term, cache, api_key)
        if fr_syns:
            # use first (preferred) synonym only for the prompt
            term_pairs.append((ent.text, fr_syns[0]))

    if not term_pairs:
        return PLAIN_INSTRUCTION.format(source=source), 0

    glossary_lines = "\n".join(
        f'"{en}" -> "{fr}"' for en, fr in term_pairs
    )
    prompt = TERM_INSTRUCTION.format(
        glossary_lines=glossary_lines,
        source=source,
    )
    return prompt, len(term_pairs)


def main():
    nlp   = load_ner()
    cache = load_cache(CACHE_PATH)

    for split in SPLITS:
        en_path  = split["en"]
        msg_path = split["msgs"]
        out_path = split["output"]

        if not en_path.exists():
            print(f"Skipping {en_path} — not found")
            continue
        if not msg_path.exists():
            print(f"Skipping {msg_path} — not found")
            continue

        print(f"\nProcessing {en_path.parent.name} split...")
        out_path.parent.mkdir(parents=True, exist_ok=True)

        total        = 0
        with_terms   = 0
        without_terms = 0

        with open(en_path, encoding="utf-8") as fen, \
             open(msg_path, encoding="utf-8") as fmsg, \
             open(out_path, "w", encoding="utf-8") as fout:

            for en_line, msg_line in zip(fen, fmsg):
                source = en_line.strip()
                record = json.loads(msg_line.strip())
                total += 1

                prompt, n_terms = build_prompt(source, nlp, cache, UMLS_API_KEY)

                # rebuild messages with new prompt, keep assistant content
                new_record = {
                    "messages": [
                        {"role": "user", "content": prompt},
                        {"role": "assistant", "content": record["messages"][1]["content"]},
                    ]
                }
                fout.write(json.dumps(new_record, ensure_ascii=False) + "\n")

                if n_terms > 0:
                    with_terms += 1
                else:
                    without_terms += 1

                # save cache + progress every 500 sentences
                if total % 500 == 0:
                    save_cache(cache, CACHE_PATH)
                    print(f"  [{total}] with_terms={with_terms} without={without_terms}")

        save_cache(cache, CACHE_PATH)
        print(f"  Done: {total} sentences | {with_terms} with terms | {without_terms} plain")
        print(f"  Output: {out_path}")

    print(f"\nFinal cache size: {len(cache)} entries")


if __name__ == "__main__":
    main()