#!/usr/bin/env python3
"""Compute and save detection scoring summary for all three MoE models."""

import json
import orjson
from pathlib import Path

RESULTS_DIR = Path("/workspace/results")
RUNS = ["kmeans-full", "spectral-full", "ot-emd-full"]

results = {}
for run in RUNS:
    scores_dir = RESULTS_DIR / run / "scores" / "detection"
    if not scores_dir.exists():
        print(f"{run}: no scores found, skipping")
        continue

    rows = []
    for f in scores_dir.glob("*.txt"):
        rows.extend(orjson.loads(f.read_bytes()))

    total = len(rows)
    tp = sum(1 for r in rows if r["prediction"] and r["activating"])
    tn = sum(1 for r in rows if not r["prediction"] and not r["activating"])
    fp = sum(1 for r in rows if r["prediction"] and not r["activating"])
    fn = sum(1 for r in rows if not r["prediction"] and r["activating"])

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    bal_acc   = ((tp / (tp + fn) if (tp + fn) > 0 else 0) + (tn / (tn + fp) if (tn + fp) > 0 else 0)) / 2

    results[run] = dict(
        total=total,
        balanced_acc=round(bal_acc, 4),
        f1=round(f1, 4),
        precision=round(precision, 4),
        recall=round(recall, 4),
        tp=tp, tn=tn, fp=fp, fn=fn,
    )

lines = []
lines.append("Detection Scoring Results - MoE Expert Interpretability")
lines.append("=" * 70)
lines.append(f"Model            : facebook/opt-1.3b")
lines.append(f"Explainer        : google/gemma-3-4b-it")
lines.append(f"Tokens cached    : 5,000,000")
lines.append(f"Experts scored   : 256")
lines.append(f"Dataset          : EleutherAI/SmolLM2-135M-10B (train[:10000])")
lines.append("")
lines.append(f"{'Model':<20} {'Bal Acc':>8} {'F1':>8} {'Precision':>10} {'Recall':>8} {'Total':>8}")
lines.append("-" * 70)
for run, m in results.items():
    lines.append(
        f"{run:<20} {m['balanced_acc']:>8.4f} {m['f1']:>8.4f} {m['precision']:>10.4f} {m['recall']:>8.4f} {m['total']:>8}"
    )
lines.append("")
lines.append("Confusion Matrices")
lines.append("-" * 70)
for run, m in results.items():
    lines.append(f"\n{run}:")
    lines.append(f"  TP={m['tp']:>7}  FP={m['fp']:>7}")
    lines.append(f"  FN={m['fn']:>7}  TN={m['tn']:>7}")

output = "\n".join(lines)
print(output)

out_txt = RESULTS_DIR / "summary_table.txt"
out_json = RESULTS_DIR / "summary.json"

out_txt.write_text(output)
with open(out_json, "w") as f:
    json.dump(results, f, indent=2)

print(f"\nSaved to {out_txt}")
print(f"Saved to {out_json}")
