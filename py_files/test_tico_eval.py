from comet import download_model, load_from_checkpoint

src = open("data/processed/test/test.en").read().splitlines()
ref = open("data/processed/test/test.fr").read().splitlines()
hyp_files = {
    "sft_tower":        "models/sft_tower_tico.txt",
    "sft_qwen":         "models/sft_qwen_tico.txt",
    "sft_ministral":    "models/sft_ministral_tico.txt",
    "sft_terms_tower":     "models/sft_terms_tower_tico.txt",
    "sft_terms_qwen":      "models/sft_terms_qwen_tico.txt",
    "sft_terms_ministral": "models/sft_terms_ministral_tico.txt",
    "dpo_tower":        "models/dpo_tower_tico.txt",
    "dpo_qwen":         "models/dpo_qwen_tico.txt",
    "dpo_ministral":    "models/dpo_ministral_tico.txt",
}

model = load_from_checkpoint(download_model("Unbabel/wmt22-comet-da"))
for name, path in hyp_files.items():
    hyp = open(path).read().splitlines()
    data = [{"src": s, "mt": h, "ref": r} for s, h, r in zip(src, hyp, ref)]
    out = model.predict(data, batch_size=15, gpus=1)
    print(f"{name}: {out['system_score']:.3f}")