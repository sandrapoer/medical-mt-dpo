# sacrebleu reference -l en-de -i test
# sacrebleu data/processed/test_emea/test.fr -l en-fr -i ministralMT.txt -m bleu chrf
# comet but wihtout significance because broken
# from github unbabel comet

import json
from comet import download_model, load_from_checkpoint

'''
# EMEA test set
src = open("data/processed/test_emea/test.en").read().splitlines()
ref = open("data/processed/test_emea/test.fr").read().splitlines()
hyp_files = {"tower": "towerMT.txt", "qwen": "qwenMT.txt", "ministral": "ministralMT.txt"}
'''

# TICO19 test set
src = open("data/processed/test/test.en").read().splitlines()
ref = open("data/processed/test/test.fr").read().splitlines()
hyp_files = {"tower": "models/baseline_tower_tico.txt", "qwen": "models/baseline_qwen_tico.txt", "ministral": "models/baseline_ministral_tico.txt"}
model = load_from_checkpoint(download_model("Unbabel/wmt22-comet-da"))

for name, path in hyp_files.items():
    hyp = open(path).read().splitlines()
    data = [{"src": s, "mt": h, "ref": r} for s, h, r in zip(src, hyp, ref)]
    out = model.predict(data, batch_size=32, gpus=1)
    print(f"{name}: {out['system_score']:.3f}")