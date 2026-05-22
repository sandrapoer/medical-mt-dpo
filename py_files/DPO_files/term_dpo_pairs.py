import json
import os
import spacy
from pathlib import Path
from dotenv import load_dotenv

load_dotenv("/home/c2410843006/master-thesis/.env")

DATA_PROCESSED_DIR = Path(os.getenv("DATA_PROCESSED_DIR"))
GLOSSARY_DIR       = Path(os.getenv("GLOSSARY_DIR"))

HYPOTHESES_PATH = DATA_PROCESSED_DIR / "dpo/hypotheses.jsonl"
OUTPUT_PATH     = DATA_PROCESSED_DIR / "dpo/dpo_term_pairs.jsonl"
GLOSSARY_PATH   = GLOSSARY_DIR / "English_French_medglossaries.tsv"


def load_glossary(path: Path) -> dict:
    glossary = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            parts = line.split("\t")
            if len(parts) >= 2:
                en_term = parts[0].strip().strip("'").lstrip("0123456789").strip().lower()
                fr_term = parts[1].strip().strip("'").strip().lower()
                if en_term and fr_term:
                    glossary[en_term] = fr_term
    print(f"Glossary loaded: {len(glossary)} EN-FR pairs")
    return glossary


def load_ner():
    for model in ["en_ner_bc5cdr_md", "en_core_sci_md", "en_core_sci_sm"]:
        try:
            nlp = spacy.load(model)
            print(f"Loaded spaCy model: {model}")
            return nlp
        except OSError:
            continue
    raise RuntimeError(
        "No scispaCy model found. Install one:\n"
        "  ~/master-thesis/venv/bin/python -m pip install scispacy\n"
        "  ~/master-thesis/venv/bin/python -m pip install "
        "https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.4/en_ner_bc5cdr_md-0.5.4.tar.gz"
    )


def extract_en_terms(source: str, nlp, glossary: dict) -> list:
    """Extract medical entities from EN source that exist in the glossary."""
    doc = nlp(source)
    terms = []
    for ent in doc.ents:
        term = ent.text.lower()
        if term in glossary:
            terms.append(term)
    return terms


def score_hypothesis(hypothesis: str, expected_fr_terms: list) -> int:
    """Count how many expected FR terms appear in the hypothesis."""
    hyp_lower = hypothesis.lower()
    return sum(1 for fr_term in expected_fr_terms if fr_term in hyp_lower)


def main():
    glossary = load_glossary(GLOSSARY_PATH)
    nlp = load_ner()

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    total        = 0
    kept         = 0
    skipped_zero = 0
    skipped_tie  = 0

    with open(HYPOTHESES_PATH, encoding="utf-8") as fin, \
         open(OUTPUT_PATH, "w", encoding="utf-8") as fout:

        for line in fin:
            line = line.strip()
            if not line:
                continue
            total += 1
            record     = json.loads(line)
            source     = record["source"]
            hypotheses = record["hypotheses"]

            # 1. Extract EN medical terms present in glossary
            en_terms = extract_en_terms(source, nlp, glossary)
            if not en_terms:
                skipped_zero += 1
                continue

            # 2. Get expected FR equivalents
            expected_fr = [glossary[t] for t in en_terms]

            # 3. Score each hypothesis
            scores = [score_hypothesis(h, expected_fr) for h in hypotheses]

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
                "source":         source,
                "chosen":         hypotheses[chosen_idx],
                "rejected":       hypotheses[rejected_idx],
                "chosen_score":   max_score,
                "rejected_score": min_score,
                "matched_terms":  en_terms,
                "expected_fr":    expected_fr,
            }
            fout.write(json.dumps(pair, ensure_ascii=False) + "\n")
            kept += 1

    print(f"\nDone.")
    print(f"  Total sentences:      {total}")
    print(f"  Kept (DPO pairs):     {kept}")
    print(f"  Skipped (no terms):   {skipped_zero}")
    print(f"  Skipped (tied score): {skipped_tie}")
    print(f"  Output: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()