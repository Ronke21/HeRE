"""
Evaluate silver denoising quality against manually annotated silver examples.

Usage:
    python -m scripts.analysis.eval_silver_validation \
        [--input outputs/silver_validation/silver_sample_500clean_500noise_manually_annotated200.xlsx] \
        [--output outputs/silver_validation/silver_validation_summary.md]
"""

import argparse
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, confusion_matrix


MODELS = [
    ("agree2",            "agree2__pred"),
    ("vote_all",          "vote_all__pred"),
    ("vote7",             "vote7__pred"),
    ("agree4",            "agree4__pred"),
    ("gemma4_31b",        "opensource_llm__gemma4_31b__pred"),
    ("gemma3_27b",        "opensource_llm__gemma3_27b__pred"),
    ("dictalm3",          "opensource_llm__dictalm3__pred"),
    ("dictalm24b_nli",    "finetuned_nli__dictalm24b__pred"),
    ("aya32b_nli",        "finetuned_nli__aya32b__pred"),
    ("gemma2_9b_nli",     "finetuned_nli__gemma2_9b__pred"),
    ("xlmroberta_nli",    "encoder_nli__xlmroberta__pred"),
    ("neodictabert_nli",  "encoder_nli__neodictabert__pred"),
    ("neodictabert_rc",   "cross_rc__neodictabert_k3__pred"),
    ("mmbert_rc",         "cross_rc__mmbert_k5__pred"),
]


def compute_metrics(y_true, y_pred):
    return {
        "f1":       f1_score(y_true, y_pred, zero_division=0),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall":   recall_score(y_true, y_pred, zero_division=0),
        "accuracy": accuracy_score(y_true, y_pred),
    }


def build_summary(df_ann: pd.DataFrame) -> str:
    y_true = df_ann["manual_label"].astype(int)
    n = len(df_ann)
    n_pos = (y_true == 1).sum()
    n_neg = (y_true == 0).sum()

    lines = []
    lines.append("# Silver Validation Summary\n")
    lines.append("Evaluation of denoising model performance on a manually annotated sample")
    lines.append("drawn from the silver dataset (500 agree2-clean + 500 agree2-noise, 200 annotated).\n")

    lines.append("## Annotated Sample\n")
    lines.append("| | |")
    lines.append("|---|---|")
    lines.append("| Total annotated | %d |" % n)
    lines.append("| Positive (relation present) | %d (%.1f%%) |" % (n_pos, n_pos / n * 100))
    lines.append("| Negative (relation absent) | %d (%.1f%%) |" % (n_neg, n_neg / n * 100))
    lines.append("")

    # agree2 precision breakdown
    clean = df_ann[df_ann["agree2__pred"] == 1]
    noise = df_ann[df_ann["agree2__pred"] == 0]
    lines.append("## agree2 Precision Breakdown\n")
    lines.append("| Group | n | Actual correct | Precision |")
    lines.append("|---|---|---|---|")
    lines.append("| agree2=1 (predicted clean) | %d | %d | %.1f%% |" % (
        len(clean), (clean["manual_label"] == 1).sum(), (clean["manual_label"] == 1).mean() * 100))
    lines.append("| agree2=0 (predicted noise) | %d | %d | %.1f%% |" % (
        len(noise), (noise["manual_label"] == 0).sum(), (noise["manual_label"] == 0).mean() * 100))
    lines.append("")

    # per-model metrics
    lines.append("## Per-Model Performance\n")
    lines.append("| Model | F1 | Precision | Recall | Accuracy |")
    lines.append("|---|---|---|---|---|")
    rows = []
    for name, col in MODELS:
        if col not in df_ann.columns:
            continue
        y_pred = df_ann[col].fillna(0).astype(int)
        m = compute_metrics(y_true, y_pred)
        rows.append((name, m))
    rows.sort(key=lambda x: x[1]["f1"], reverse=True)
    for name, m in rows:
        lines.append("| %s | %.3f | %.3f | %.3f | %.3f |" % (
            name, m["f1"], m["precision"], m["recall"], m["accuracy"]))
    lines.append("")

    # predicate distribution
    lines.append("## Predicate Distribution in Annotated Sample\n")
    lines.append("| Predicate | Count | Positive | Positive rate |")
    lines.append("|---|---|---|---|")
    for pred, grp in df_ann.groupby("predicate"):
        cnt = len(grp)
        pos = (grp["manual_label"] == 1).sum()
        lines.append("| %s | %d | %d | %.0f%% |" % (pred, cnt, pos, pos / cnt * 100))
    lines.append("")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",  default="outputs/silver_validation/silver_sample_500clean_500noise_manually_annotated200.xlsx")
    parser.add_argument("--output", default="outputs/silver_validation/silver_validation_summary.md")
    args = parser.parse_args()

    df = pd.read_excel(args.input)
    df_ann = df[df["manual_label"].notna()].copy()
    print("Loaded %d annotated rows from %s" % (len(df_ann), args.input))

    summary = build_summary(df_ann)

    with open(args.output, "w", encoding="utf-8") as f:
        f.write(summary)
    print("Summary written to %s" % args.output)

    # also print to stdout
    print()
    y_true = df_ann["manual_label"].astype(int)
    print("%-22s %6s %6s %6s %6s" % ("Model", "F1", "Prec", "Rec", "Acc"))
    print("-" * 52)
    rows = []
    for name, col in MODELS:
        if col not in df_ann.columns:
            continue
        y_pred = df_ann[col].fillna(0).astype(int)
        m = compute_metrics(y_true, y_pred)
        rows.append((name, m))
    rows.sort(key=lambda x: x[1]["f1"], reverse=True)
    for name, m in rows:
        print("%-22s %6.3f %6.3f %6.3f %6.3f" % (name, m["f1"], m["precision"], m["recall"], m["accuracy"]))


if __name__ == "__main__":
    main()
