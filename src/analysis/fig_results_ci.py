"""Figure: best micro-F1 per model on the gold validation set with 95% bootstrap
CIs, grouped by strategy, against the two CROCODILE baselines. Values are the
Table 2 entries; CIs come from stage0_results (S2/S4/S5), the S1 fixed-threshold
file, and a bootstrap over the S3 ensemble gold predictions (adjudicated labels).
Output: paper/fig_results_ci.pdf"""
import sys; sys.path.insert(0, ".")
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from post_rebuttal_and_camera_ready.analysis.stage0_ci_and_macro import _adjudicated_labels, JOIN_KEYS, bootstrap_f1
R = "post_rebuttal_and_camera_ready/analysis/stage0_results/"
ci = pd.read_csv(R + "bootstrap_ci_all_configs.csv")
def ci_of(cfg):
    r = ci[ci.config == cfg].iloc[0]; return float(r.micro_f1), float(r.ci_lo), float(r.ci_hi)
s1 = pd.read_csv(R + "encoder_nli_v3_fixedthresh.csv")
def s1_of(m):
    r = s1[s1.model == m].sort_values("f1", ascending=False).iloc[0]; return float(r.f1), float(r.ci_lo), float(r.ci_hi)
labels = _adjudicated_labels()
def s3_of(path):
    d = pd.read_csv(path, encoding="utf-8-sig")
    key = d[JOIN_KEYS].astype(str).agg("||".join, axis=1); y = key.map(labels).fillna(d.relation_present).astype(int).to_numpy()
    p = d.predicted_relation_present.astype(int).to_numpy(); f, lo, hi = bootstrap_f1(y, p)
    return float(f), float(lo), float(hi)
G = "post_rebuttal_and_camera_ready/gold_benchmark_v3/cross_train_rc/"
TABLE2_S3 = {G + "me5large_k7/me5large/k7/steps8000_lr2e-5/gold_classified.csv": 0.663, G + "xlmroberta_k3/xlmroberta/k3/steps8000_lr2e-5/gold_classified.csv": 0.718,
             G + "mmbert_k10/mmbert/k10/steps8000_lr2e-5/gold_classified.csv": 0.733, G + "neodictabert_k7/neodictabert/k7/steps16000_lr2e-5/gold_classified.csv": 0.742}
rows = [  # (strategy, label, (f1, lo, hi))
 ("S1: NLI encoders", "AlephBERT", s1_of("alephbert")), ("S1: NLI encoders", "mE5-large", s1_of("me5large")), ("S1: NLI encoders", "mmBERT", s1_of("mmbert")),
 ("S1: NLI encoders", "XLM-R-large", s1_of("xlmroberta")), ("S1: NLI encoders", "NeoDictaBERT", s1_of("neodictabert")), ("S1: NLI encoders", "XLM-R-XL", s1_of("xlmrobertaxl")),
 ("S2: fine-tuned LLM NLI", "DictaLM-3.0-1.7B", ci_of("nli_clean_dictalm3_1_7b_template") if (ci.config == "nli_clean_dictalm3_1_7b_template").any() else None),
 ("S2: fine-tuned LLM NLI", "Gemma-4-26B-A4B", None), ("S2: fine-tuned LLM NLI", "Gemma3-12B-it", ci_of("nli_clean_gemma3_12b_template")),
 ("S2: fine-tuned LLM NLI", "Gemma-3-27B-it", ci_of("nli_clean_gemma3_27b_template")), ("S2: fine-tuned LLM NLI", "DictaLM-3.0-24B-Base", ci_of("nli_clean_dictalm24b_base_v2_template")),
 ("S2: fine-tuned LLM NLI", "Aya-Expanse-32b", ci_of("nli_clean_aya32b_v2_template")), ("S2: fine-tuned LLM NLI", "Gemma-2-9B", ci_of("nli_clean_gemma2_9b_template")),
 ("S2: fine-tuned LLM NLI", "Gemma-4-31B", ci_of("nli_clean_gemma4_31b_v2_template")), ("S2: fine-tuned LLM NLI", "DictaLM-3.0-24B", ci_of("nli_clean_dictalm24b_v2_template")),
 ("S2: fine-tuned LLM NLI", "Qwen3.5-35B-A3B", ci_of("nli_clean_qwen35base_v2_template")),
 ("S3: cross-trained", "mE5-large", s3_of(G + "me5large_k7/me5large/k7/steps8000_lr2e-5/gold_classified.csv")),
 ("S3: cross-trained", "XLM-R-large", s3_of(G + "xlmroberta_k3/xlmroberta/k3/steps8000_lr2e-5/gold_classified.csv")),
 ("S3: cross-trained", "mmBERT", s3_of(G + "mmbert_k10/mmbert/k10/steps8000_lr2e-5/gold_classified.csv")),
 ("S3: cross-trained", "NeoDictaBERT", s3_of(G + "neodictabert_k3_dgx/neodictabert/k3/steps16000_lr2e-5/gold_classified.csv")),
 ("S4: open LLMs, few-shot", "DictaLM-3.0-24B-Base", ci_of("llm_clean_dictalm3_he_triplet_5s_nc")), ("S4: open LLMs, few-shot", "Aya-Expanse-32b", ci_of("llm_clean_cohere_aya32b_he_template_2s_nc")),
 ("S4: open LLMs, few-shot", "Gemma-3-27B-it", ci_of("llm_clean_gemma3_he_template_5s_nc")), ("S4: open LLMs, few-shot", "Gemma-4-26B-A4B-it", ci_of("llm_clean_gemma4_it_en_template_2s_nc")),
 ("S4: open LLMs, few-shot", "Gemma-4-31B-it", ci_of("llm_clean_gemma4_31b_it_he_template_2s_nc")),
 ("S5: API LLMs", "Claude Haiku 4.5", ci_of("llm_clean_claude_haiku45_he_triplet_5s_nc")), ("S5: API LLMs", "Gemini 3 Flash", ci_of("llm_clean_gemini3_flash_en_triplet_5s_nc")), ("S5: API LLMs", "GPT-5.4", ci_of("llm_clean_gpt54_he_triplet_5s_nc")),
]
# S2 rows lacking a config in the CI file: fill from the CI table by best-F1 lookup
for i, (s, l, v) in enumerate(rows):
    if v is None:
        pat = {"DictaLM-3.0-1.7B": r"dictalm.*(1_7|17b|1\.7)", "Gemma-4-26B-A4B": "gemma4_26"}[l]
        sub = ci[ci.config.str.contains(pat, regex=True)]; assert len(sub), l.sort_values("micro_f1", ascending=False)
        rows[i] = (s, l, (float(sub.iloc[0].micro_f1), float(sub.iloc[0].ci_lo), float(sub.iloc[0].ci_hi)))
for s, l, (f, lo, hi) in rows: print(f"{s:26s} {l:22s} {f:.3f} [{lo}, {hi}]")
# ---- draw ----
plt.rcParams.update({"font.size": 7, "font.family": "serif", "pdf.fonttype": 42, "ps.fonttype": 42})
strategies = list(dict.fromkeys(s for s, _, _ in rows))
colors = {"S1: NLI encoders": "#4c72b0", "S2: fine-tuned LLM NLI": "#dd8452", "S3: cross-trained": "#55a868", "S4: open LLMs, few-shot": "#c44e52", "S5: API LLMs": "#8172b3"}
fig, ax = plt.subplots(figsize=(3.2, 4.4))
y = 0; yt, yl = [], []
for s in strategies:
    ax.text(0.30, y, s, fontsize=6.5, fontweight="bold", va="center", color=colors[s]); y -= 1
    for s2, l, (f, lo, hi) in rows:
        if s2 != s: continue
        ax.plot([lo, hi], [y, y], color=colors[s], lw=1); ax.plot(f, y, "o", color=colors[s], ms=3.2)
        yt.append(y); yl.append(l); y -= 1
    y -= 0.4
for x, lab in [(0.406, "CROCODILE\nas published"), (0.784, "CROCODILE\ntuned")]:
    ax.axvline(x, color="0.35", ls="--", lw=0.8); ax.text(x, y + 0.3, lab, fontsize=5.5, ha="center", va="top", color="0.35")
ax.set_yticks(yt); ax.set_yticklabels(yl); ax.set_xlim(0.33, 0.97); ax.set_ylim(y - 1.2, 1); ax.set_xlabel("Binary F1 (95% bootstrap CI)")
ax.tick_params(axis="y", length=0); ax.spines[["top", "right", "left"]].set_visible(False); ax.grid(axis="x", color="0.9", lw=0.6)
fig.tight_layout(pad=0.3); fig.savefig("post_rebuttal_and_camera_ready/paper/fig_results_ci.pdf"); print("wrote fig_results_ci.pdf")
