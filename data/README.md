# Data

| Resource | Where | Rows |
|---|---|---|
| Gold validation set (adjudicated, dual-annotated) | `gold/gold_validation.csv` and HF `gold/validation.parquet` | 500 |
| Gold test set (held out, dual-annotated) | `gold/gold_test.csv` and HF `gold/test.parquet` | 500 |
| HeRE silver (published `agree3` label + all signals) | HF `silver/here_silver.parquet` | 1,491,070 |
| Full per-signal release (every silver row, 10 signals, 4 consensus rules) | HF `silver/silver_all_signals.parquet` | 2,564,034 |
| HeRE-KG (evidence-linked edges) | HF `kg/here_kg_edges.tsv.gz` | 1,080,605 |

HF = https://huggingface.co/datasets/ronke21/HeRE

Gold columns: `item_id, docid, title, text, subject, predicate, object, label, annotator1, annotator2, notes, word_count` (`selection_reason` is `validation` in the validation set and `new_predicate` 203, `depth_boost` 97, `silver_validation_200` 200 in the test set). `annotator1` and `annotator2` are the two annotators' labels (annotator 2 annotated blind); `label` is the adjudicated label used in every table of the paper (identical to `annotator1`, whose column absorbed the adjudication decisions). Inter-annotator agreement before adjudication: Cohen's kappa 0.809 (validation), 0.757 (test).

All 500 test rows, which were drawn from the silver corpus, are withheld from every silver and KG file (`src/analysis/build_here_kg.py::test_row_mask`). Do not train on them.

Text: Hebrew Wikipedia (CC BY-SA 4.0). Triples: Wikidata (CC0). Annotation guidelines: `gold/annotation_guidelines.md`.
