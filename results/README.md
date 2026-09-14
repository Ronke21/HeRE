# Results

Every number in the paper's tables and figures comes from a file here.

| File | Paper location | Content |
|---|---|---|
| `main/bootstrap_ci_all_configs.csv` | Table 2, Figure 2, Tables 8/10/11 | micro-F1 with 95% bootstrap CI and macro-F1 for every Strategy 2/4/5 configuration on the gold validation set (adjudicated labels) |
| `main/encoder_nli_v3_fixedthresh.csv` | Table 2, Table 7 | Strategy 1 encoders, fixed-threshold protocol, all hypothesis/formula combinations |
| `main/s3_grid_final.csv` | Table 2, Table 9 | Strategy 3 grid (4 encoders × K ∈ {3,5,7,10}) on the final adjudicated labels; `s3_grid_adjudicated.csv` is the earlier scoring on mid-adjudication labels and is superseded |
| `main/crocodile_baseline.csv` | Table 2, §6 | the CROCODILE XLM-R XNLI filter as published and tuned |
| `main/paired_bootstrap_top.json` | §6 | paired bootstrap between the best configuration and the next five |
| `main/test_set_all_signals.csv`, `main/test_set_subsets.csv` | Tables 4 and 5 | every deployed signal on the held-out test set; Gemma-4-31B-it by test subset |
| `main/agree3_test_quality.csv` | §7, Table 16 | coverage, precision, recall and F1 of the published `agree3` label on the test set and its subsets |
| `main/per_predicate_best_config.csv`, `main/length_bucketed_f1.csv` | Tables 13 and 14 | per-predicate and length-bucketed F1 |
| `main/table2_v3.csv` | Table 1 | dataset statistics |
| `kge_ablation/` | Table 3 | TransE link prediction, four graph variants × three seeds (`seed_summary.csv`) |
| `here_kg/here_kg_stats.md` | §7 | HeRE-KG statistics (gold-test rows withheld); `ego_edges.csv` backs Figure 3 |
| `re_pilot/` | not in the paper | mmBERT fine-tuning pilot on denoised vs raw data |
| `s3_v1text_diagnostic.txt` | not in the paper | Strategy 3 rerun on the original (v1) text, showing the text version is not what changed the Strategy 3 scores |
