"""
Regenerate 07_predicate_relation_present_stacked.png with English predicate labels.
Output: dataset_statistics/gold/07_predicate_relation_present_stacked.png  (overwrite)
"""

from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

ROOT     = Path(__file__).resolve().parent.parent.parent
GOLD_CSV = ROOT / "data/crocodile_heb25_gold_500.csv"
OUT_PATH = ROOT / "dataset_statistics/gold/07_predicate_relation_present_stacked.png"

PRED_EN = {
    "מדינה":               "Country",
    "מופע של":             "Instance of",
    "יחידה מנהלית":        "Administrative unit",
    "גובל עם":             "Borders",
    "שפה רשמית":           "Official language",
    "תת-קבוצה של":         "Subclass of",
    "חלוקה משנית":         "Admin. subdivision",
    "שפה שבשימוש":         "Language in use",
    "עיסוק":               "Occupation",
    "אזרחות":              "Citizenship",
    "יחסים דיפלומטיים":    "Diplomatic relations",
    "חלק מתוך":            "Part of",
    "אמן מבצע":            "Performer",
    "ההפך מ־":             "Opposite of",
    "אחים ואחיות":          "Siblings",
    "בירה של":             "Capital of",
    "מקום מוצא":           "Place of origin",
    "מקום לידה":           "Place of birth",
    "עיר בירה":            "Capital city",
    "יבשת":               "Continent",
}

gold = pd.read_csv(GOLD_CSV)
gold["predicate_en"] = gold["predicate"].map(PRED_EN).fillna(gold["predicate"])

top15_heb = gold["predicate"].value_counts().head(15).index.tolist()
top15_en  = [PRED_EN.get(p, p) for p in top15_heb]

ct = (
    gold[gold["predicate"].isin(top15_heb)]
    .assign(predicate_en=lambda d: d["predicate"].map(PRED_EN).fillna(d["predicate"]))
    .groupby(["predicate_en", "relation_present"])
    .size()
    .unstack(fill_value=0)
    .reindex(columns=[0, 1])
)
ct_norm = ct.div(ct.sum(axis=1), axis=0) * 100
ct_norm = ct_norm.loc[ct_norm[1].sort_values(ascending=False).index]

fig, ax = plt.subplots(figsize=(11, 5))
x = np.arange(len(ct_norm))
w = 0.6
ax.bar(x, ct_norm[0], w, label="Absent (0)", color="#d62728", alpha=0.8, edgecolor="white")
ax.bar(x, ct_norm[1], w, bottom=ct_norm[0], label="Present (1)", color="#2ca02c", alpha=0.8, edgecolor="white")
ax.set_xticks(x)
ax.set_xticklabels(ct_norm.index.tolist(), fontsize=9, rotation=35, ha="right")
ax.set_ylabel("Percentage (%)")
ax.set_ylim(0, 118)
ax.yaxis.set_major_formatter(mticker.PercentFormatter())
ax.set_title("Relation-Present Rate by Predicate  (Gold, top 15)", fontsize=11)
ax.legend(loc="upper right")
for i, (pred, row) in enumerate(ct_norm.iterrows()):
    ax.text(i, 103, f"n={int(ct.loc[pred].sum())}", ha="center", va="bottom", fontsize=7)

fig.tight_layout()
fig.savefig(OUT_PATH, dpi=150, bbox_inches="tight")
print(f"Saved → {OUT_PATH}")
