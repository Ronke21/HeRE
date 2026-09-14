"""
Applications §1 figure — ego-network of one entity in HeRE-KG.

Draws the KG neighborhood of a chosen entity (default: David Ben-Gurion):
central node, top-N neighbors by confidence (deduped by predicate+object),
predicate-labeled edges, edge width ∝ mean signal score. Also dumps the
underlying rows (with docid provenance) to CSV so the paper caption can
quote the supporting passage of any edge.

Matplotlib renders Hebrew glyphs (DejaVu Sans) but not RTL ordering, so
labels are visually reversed for display only; the CSV keeps logical order.

Usage (env with matplotlib+networkx, e.g. heb_relation_extraction):
    python post_rebuttal_and_camera_ready/analysis/kg_ego_figure.py \
        [--entity "..."] [--top 10]
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd

HERE = Path(__file__).resolve().parent
EDGES = HERE / "here_kg" / "here_kg_edges.tsv.gz"
OUTD = HERE / "here_kg"


def rtl(s: str) -> str:
    """Visual reversal for matplotlib (display only)."""
    return str(s)[::-1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--entity", default="דוד בן-גוריון")
    ap.add_argument("--top", type=int, default=10)
    args = ap.parse_args()

    df = pd.read_csv(EDGES, sep="\t")
    ego = df[(df.subject == args.entity) | (df.object == args.entity)].copy()
    if ego.empty:
        raise SystemExit(f"entity not found: {args.entity}")

    # one edge per (predicate, neighbor), keep the best-evidenced mention
    ego["neighbor"] = ego.apply(
        lambda r: r["object"] if r["subject"] == args.entity else r["subject"], axis=1)
    ego = (ego.sort_values("mean_score", ascending=False)
              .drop_duplicates(["predicate", "neighbor"]))
    top = ego.head(args.top)
    top.to_csv(OUTD / "ego_edges.csv", index=False)

    def wrap(label, width=8):
        """Wrap at spaces, then reverse each line for matplotlib's LTR rendering."""
        words, lines, cur = str(label).split(), [], ""
        for w in words:
            if cur and len(cur) + 1 + len(w) > width:
                lines.append(cur); cur = w
            else:
                cur = (cur + " " + w).strip()
        lines.append(cur)
        return "\n".join(l[::-1] for l in lines)

    G = nx.Graph()
    G.add_node(args.entity)
    for _, r in top.iterrows():
        G.add_edge(args.entity, r["neighbor"],
                   predicate=r["predicate"], score=r["mean_score"])

    # ego in the centre, neighbours evenly on a circle: no overlaps, predictable
    others = [n for n in G if n != args.entity]
    pos = {args.entity: (0.0, 0.0)}
    for i, n in enumerate(others):
        ang = 2 * math.pi * i / len(others) + math.pi / 2
        pos[n] = (2.6 * math.cos(ang), 2.6 * math.sin(ang))

    # sized for a single column: large fonts, labels inside the circles
    fig, ax = plt.subplots(figsize=(6.4, 6.4))
    widths = [1.5 + 4.0 * G[u][v]["score"] for u, v in G.edges()]
    nx.draw_networkx_edges(G, pos, ax=ax, width=widths, edge_color="#8aa2c0")
    nx.draw_networkx_nodes(G, pos, nodelist=[args.entity], node_color="#f4a261",
                           edgecolors="#c1121f", linewidths=1.5, node_size=6200, ax=ax)
    nx.draw_networkx_nodes(G, pos, nodelist=others, node_color="#e8eef7",
                           edgecolors="#8aa2c0", linewidths=1.2, node_size=5200, ax=ax)
    nx.draw_networkx_labels(G, pos, {n: wrap(n) for n in G}, font_size=11,
                            font_weight="bold", ax=ax)
    nx.draw_networkx_edge_labels(
        G, pos, {(u, v): wrap(G[u][v]["predicate"], 14) for u, v in G.edges()},
        font_size=10, font_color="#1d3557", label_pos=0.55,
        bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.9), ax=ax)
    ax.set_xlim(-3.5, 3.5); ax.set_ylim(-3.5, 3.5)
    ax.axis("off")
    fig.tight_layout(pad=0.2)
    for ext in ("pdf", "png"):
        fig.savefig(OUTD / f"ego_network.{ext}", dpi=200)
    print(f"wrote ego_network.pdf/.png + ego_edges.csv → {OUTD}")
    print(top[["subject", "predicate", "object", "docid", "mean_score"]].to_string(index=False))


if __name__ == "__main__":
    main()
