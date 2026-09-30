# HeRE: Hebrew Relation Extraction benchmark

Code, gold data and result files for **HeRE: A Novel Benchmark for Hebrew Relation Extraction via LLM-Guided Denoising of Knowledge Graph Alignments** (Keinan, Cohen, Tsarfaty; AACL 2026).

HeRE turns the noisy distant-supervision alignment between Hebrew Wikipedia and Wikidata (3.1M candidate passage-triple tuples, built with CROCODILE) into a relation-extraction resource by *denoising* it: five families of denoisers are compared on a 500-example dual-annotated gold validation set, the best open-weights configurations are applied to the full corpus, and the unanimous vote of three of them (`agree3`) is released as a 1.49M-row silver dataset together with the raw score of every deployed signal, a held-out gold test set, and a provenance-grounded knowledge graph (HeRE-KG).

The camera-ready paper is in `paper/HeRE_AACL2026.pdf`. Datasets are on Hugging Face: **https://huggingface.co/datasets/ronke21/HeRE** (see `data/README.md`).

## Headline numbers (gold validation set, binary F1, adjudicated labels)

| Strategy | Best model | F1 |
|---|---|---|
| 1 Fine-tuned NLI encoders | XLM-R-XL | 0.779 |
| 2 Fine-tuned LLM NLI (LoRA) | Qwen3.5-35B-A3B-Base | 0.873 |
| 3 Cross-trained predicate classifiers | NeoDictaBERT (K=3) | 0.751 |
| 4 Open-weights LLMs, few-shot | Gemma-4-31B-it | **0.915** |
| 5 API LLMs, few-shot | GPT-5.4 | 0.921 |
| Baseline: CROCODILE's XLM-R XNLI filter | as published / tuned | 0.406 / 0.784 |

Held-out test set (Gemma-4-31B-it): 0.844 overall, 0.944 on predicates seen in validation, 0.776 on 203 unseen predicates. Link prediction on HeRE-KG (TransE, three seeds): the denoised graph beats a size-matched random control by 24% relative MRR. Full result files are indexed in `results/README.md`. Published `agree3` label on the test rows it covers (66.8%): precision 0.899, recall 0.934, F1 0.916. Mean 95% bootstrap CI half-width on 500 examples: ±0.022; the top five configurations are statistically indistinguishable.

## Layout

```
data/gold/            gold validation + test CSVs, annotation guidelines
src/prepare_data/     text cleaning (text_cleaning.py), hypothesis generation, gold/silver preparation (v3)
src/denoise_gold/     the five strategies evaluated on the gold set:
                        clean_encoder_NLI.py            Strategy 1
                        clean_finetuned_llm_NLI.py      Strategy 2
                        clean_cross_train_rc.py         Strategy 3
                        clean_dataset_with_opensource_llm.py  Strategy 4 (prompts + hand-written demonstrations inside)
                        clean_dataset_with_api_llm.py   Strategy 5 (OpenRouter; needs OPENROUTER_API_KEY)
src/denoise_silver/   the same denoisers applied to the 2.56M-row silver corpus, resumable, one float per line
src/pipeline/         campaign driver: units.py (registry of work units), shim.py (path rebinding), watcher.py
                      (SLURM/A100 lanes), build_silver_v3.py (merge into silver_all_cleaned.parquet), slurm_unit.sbatch
src/nli_finetuning/   HebNLI fine-tuning of the encoders and LoRA decoders (depends on the mrl_eval package of
                      https://github.com/Ronke21/HEntailment; hyperparameters are in the paper, Appendix D)
src/analysis/         bootstrap CIs and macro-F1 (stage0_ci_and_macro.py), paired bootstrap, per-predicate and
                      length-bucketed F1, held-out test evaluation (test_set_all_signals.py, agree3_test_quality.py),
                      CROCODILE baseline, HeRE-KG builder (build_here_kg.py), TransE ablation (kge_ablation.py),
                      RE fine-tuning pilot, ego-network figure, dataset statistics, HF release builder
results/main/         every number in the paper's tables: per-configuration F1 with CIs, macro-F1, S3 grid, test-set
                      signals and subsets, agree3 quality, per-predicate and length buckets, baseline
results/kge_ablation/ TransE link prediction, 4 graph variants x 3 seeds (+ seed_summary.csv)
results/here_kg/      HeRE-KG statistics and the ego-network edge list; figures/ has the figure
results/re_pilot/     mmBERT fine-tuning pilot on denoised vs raw data
```

## Fine-tuned models

All HeRE resources are grouped in the Hub collection https://huggingface.co/collections/ronke21/here-hebrew-relation-extraction-6aa82bf21db8a9c955cf1cd1. The HebNLI-fine-tuned denoisers behind the released signals and the per-strategy winners:

| Model | Role in HeRE | Repo |
|---|---|---|
| DictaLM-3.0-24B (Thinking) HebNLI LoRA | Strategy 2; one of the three `agree3` models behind the published label | https://huggingface.co/ronke21/hebnli-dictalm-3.0-24b-lora |
| NeoDictaBERT HebNLI | Strategy 1; deployed signal `encoder_nli__neodictabert` | https://huggingface.co/ronke21/hebnli-neodictabert |
| XLM-RoBERTa-large HebNLI | Strategy 1; deployed signal `encoder_nli__xlmroberta` | https://huggingface.co/ronke21/hebnli-xlm-roberta-large |
| XLM-RoBERTa-XL HebNLI | Strategy 1 best (F1 0.779) | https://huggingface.co/ronke21/hebnli-xlm-roberta-xl |
| Qwen3.5-35B-A3B-Base HebNLI LoRA | Strategy 2 best (F1 0.873) | https://huggingface.co/ronke21/hebnli-qwen3.5-35b-a3b-lora |
| Gemma-4-31B HebNLI LoRA | Strategy 2 (F1 0.849) | https://huggingface.co/ronke21/hebnli-gemma-4-31b-lora |
| AlephBERT HebNLI | Strategy 1 (F1 0.618) | https://huggingface.co/ronke21/hebnli-alephbert |

The Strategy 4 signals (Gemma-4-31B-it, Gemma-3-27B-it, DictaLM-3.0-24B-Base) are prompted public models; prompts are in the code repository.

## Reproducing

1. **Candidate corpus.** Run CROCODILE (https://github.com/Babelscape/crocodile) on the Hebrew Wikipedia dump and Wikidata; the output is the 3.12M-row `crocodile_heb25_full_dataset` CSV expected by `src/prepare_data/`.
2. **Preparation.** `src/prepare_data/text_cleaning.py` (markup removal, entity-field normalisation) and `prepare_v3_data.py` produce the gold and silver inputs with the three hypothesis variants (basic, template, LLM).
3. **NLI backbones.** `src/nli_finetuning/` fine-tunes the encoders (batch 64, 8,000 steps) and LoRA decoders (rank 16, batch 16) on HebNLI.
4. **Gold benchmark.** One script per strategy in `src/denoise_gold/`; `src/analysis/stage0_ci_and_macro.py` turns the predictions into the tables with bootstrap CIs.
5. **Silver corpus.** `src/pipeline/watcher.py` schedules the units in `units.py` over the available GPUs; every unit appends one score per line and resumes from its own output. `build_silver_v3.py` merges the signals; `src/analysis/build_here_kg.py` builds HeRE-KG and `build_hf_release.py` the Hugging Face files, both withholding all 500 gold-test rows.
6. **Downstream.** `src/analysis/kge_ablation.py --variant {raw,denoised,control,weighted} --seed N`.

Absolute machine paths in the scripts were replaced by `/path/to/...`; set them (mainly in `src/pipeline/paths.py`, `units.py` and the checkpoint constants at the top of the denoising scripts) before running. Model checkpoints, the CROCODILE output and the silver corpus are not in this repository.

## License and citation

Code: MIT. Text data: Hebrew Wikipedia, CC BY-SA 4.0. Triples: Wikidata, CC0. See `CITATION.cff`.
