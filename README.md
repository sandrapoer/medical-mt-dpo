# EN→FR Medical Machine Translation: SFT vs DPO with Terminology-Aware Training

Master's thesis experiments for the Joint Master Multilingual Technologies  

## Overview
This repository contains the full experimental pipeline for comparing **Supervised Fine-Tuning (SFT)** and **Direct Preference Optimization (DPO)** for English-to-French medical Machine Translation (MT), with and without terminology-aware training signals.

**Research question:** To what extent does Direct Preference Optimization improve the quality of medical terminology translation compared to Supervised Fine-Tuning when evaluated using automatic metrics and terminological accuracy?

**Models evaluated:**
- `TowerInstruct-7B-v0.2`
- `Ministral-8B-Instruct-2410`
- `Qwen3-8B`


## Datasets
The Datasets used for the experiments are the EMEA v3 EN-FR parallel corpus and the TICO-19 EN-FR benchmark. EMEA is used for training, validation and testing (20k/1k/1k). The TICO-19 benchmark is reserved as a held-out test set (2.1k). 


## Training
Standard SFT fine-tuning is run on the models using EMEA as the training corpus. For a second SFT variant, glossary headers are injected into the instruction prompt ("SFT-terms"). DPO is using model-specific preference pairs.

## Zero-Shot Baseline
A untuned zero-shot baseline is run on both test-sets for further comparison of the model's performances.

## Metrics
The final systems are evaluated using BLEU, chrF and COMET-DA.