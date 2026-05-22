import json
import os
import time
import spacy
import requests
from pathlib import Path
from dotenv import load_dotenv

load_dotenv("/home/c2410843006/master-thesis/.env")

DATA_PROCESSED_DIR = Path(os.getenv("DATA_PROCESSED_DIR"))

HYPOTHESES_PATH = DATA_PROCESSED_DIR / "dpo/hypotheses.jsonl"
OUTPUT_PATH = DATA_PROCESSED_DIR / "dpo/umls_dpo_pairs.jsonl"
CACHE_PATH = DATA_PROCESSED_DIR / "dpo/umls_cache.json"

UMLS_API_KEY = os.getenv("UMLS_API_KEY")
UMLS_VERSION = "2026AA"
UMLS_BASE = "https://uts-ws.nlm.nih.gov/rest"
REQUEST_DELAY = 0.1
TEST_RUN=True
TEST_LIMIT=10



def load_cache(path: Path) -> dict:
    if path.exists():
        with open(path, encoding="utf-8") as f:
            cache = json.load(f)
        print(f"Cache loaded: {len(cache)} entries")
        return cache
    return {}


def save_cache(cache: dict, path: Path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False)


def get_cui(term: str, api_key: str) -> str | None:
    """Search UMLS for a term and return its CUI."""
    url = f"{UMLS_BASE}/search/2026AA"    
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
    url = f"{UMLS_BASE}/content/2026AA/CUI/{cui}/atoms"
    params = {
        "apiKey": api_key,
        "pageSize": 50,
    }
    fr_terms = []
    try:
        r = requests.get(url, params=params, timeout=10)
        r.raise_for_status()
        atoms = r.json().get("result", [])
        for atom in atoms:
            if atom.get("language") == "FRE":
                name = atom.get("name", "").strip().lower()
                if name:
                    fr_terms.append(name)
    except Exception as e:
        print(f"  French atom lookup failed for CUI '{cui}': {e}")
    return fr_terms


def lookup_term(term: str, cache: dict, api_key: str) -> list:
    """Return French synonyms for an EN term, using cache where possible."""
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
    raise RuntimeError(
        "No scispaCy model found.\n"
        "  ~/master-thesis/venv/bin/python -m pip install "
        "https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.4/en_core_sci_md-0.5.4.tar.gz"
    )


def score_hypothesis(hypothesis: str, fr_terms_per_entity: list[list]) -> int:
    """
    For each source entity, check if any of its French synonyms appear
    in the hypothesis. Score = number of entities with at least one match.
    """
    hyp_lower = hypothesis.lower()
    score = 0
    for fr_synonyms in fr_terms_per_entity:
        if any(syn in hyp_lower for syn in fr_synonyms):
            score += 1
    return score


def main():
    nlp   = load_ner()
    cache = load_cache(CACHE_PATH)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)

    total = 0
    kept = 0
    skipped_zero = 0
    skipped_tie = 0
    cache_saves = 0

    with open(HYPOTHESES_PATH, encoding="utf-8") as fin, \
         open(OUTPUT_PATH, "w", encoding="utf-8") as fout:

        for line in fin:
            line = line.strip()
            if not line:
                continue
            total += 1
            if TEST_RUN and total > TEST_LIMIT:
                print(f"  Test run limit reached ({TEST_LIMIT} sentences). Stopping.")
                break
            record     = json.loads(line)
            source     = record["source"]
            hypotheses = record["hypotheses"]

            # 1. NER on EN source
            doc = nlp(source)
            entities = [ent.text for ent in doc.ents]

            if not entities:
                skipped_zero += 1
                continue

            # 2. UMLS lookup → French synonyms per entity
            fr_terms_per_entity = []
            matched_entities    = []
            for ent_text in entities:
                fr_syns = lookup_term(ent_text, cache, UMLS_API_KEY)
                if fr_syns:
                    fr_terms_per_entity.append(fr_syns)
                    matched_entities.append(ent_text)

            if not fr_terms_per_entity:
                skipped_zero += 1
                continue

            # 3. Score each hypothesis
            scores = [score_hypothesis(h, fr_terms_per_entity) for h in hypotheses]

            max_score = max(scores)
            min_score = min(scores)

            # 4. Skip if all scores equal or all zero
            if max_score == 0 or max_score == min_score:
                skipped_tie += 1
                continue

            # 5. chosen = highest score, rejected = lowest score
            chosen_idx   = scores.index(max_score)
            rejected_idx = scores.index(min_score)

            pair = {
                "source":           source,
                "chosen":           hypotheses[chosen_idx],
                "rejected":         hypotheses[rejected_idx],
                "chosen_score":     max_score,
                "rejected_score":   min_score,
                "matched_entities": matched_entities,
                "fr_synonyms":      fr_terms_per_entity,
            }
            fout.write(json.dumps(pair, ensure_ascii=False) + "\n")
            kept += 1

            # Save cache every 500 sentences
            cache_saves += 1
            if cache_saves % 500 == 0:
                save_cache(cache, CACHE_PATH)
                print(f"  [{total}/20000] kept={kept} | cache saved ({len(cache)} entries)")

    # Final cache save
    save_cache(cache, CACHE_PATH)

    print(f"\nDone.")
    print(f"  Total sentences:      {total}")
    print(f"  Kept (DPO pairs):     {kept}")
    print(f"  Skipped (no terms):   {skipped_zero}")
    print(f"  Skipped (tied score): {skipped_tie}")
    print(f"  Cache entries:        {len(cache)}")
    print(f"  Output: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()