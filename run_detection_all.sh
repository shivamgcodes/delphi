#!/bin/bash
set -e

export HF_HOME=/workspace/.cache/huggingface
export HF_DATASETS_CACHE=/workspace/.cache/huggingface/datasets
export TMPDIR=/workspace/.tmp
mkdir -p /workspace/.tmp

cd /workspace

COMMON_ARGS="
  --dataset_repo EleutherAI/SmolLM2-135M-10B
  --dataset_split train[:10000]
  --scorers detection
  --max_latents 256
  --n_tokens 5000000
  --explainer_model google/gemma-3-4b-it
  --explainer_model_max_len 8192
  --max_memory 0.85
  --verbose False
"

echo "=== Running KMeans ==="
python -m delphi facebook/opt-1.3b --moe_mode \
  --moe_wrapper_path /workspace/slice/models/exp2/kmeans_13b_wrapper.pt \
  $COMMON_ARGS --name /workspace/results/kmeans-full

echo "=== Running Spectral ==="
python -m delphi facebook/opt-1.3b --moe_mode \
  --moe_wrapper_path /workspace/slice/models/exp2/spectral_13b_wrapper.pt \
  $COMMON_ARGS --name /workspace/results/spectral-full

echo "=== Running OT-EMD ==="
python -m delphi facebook/opt-1.3b --moe_mode \
  --moe_wrapper_path /workspace/slice/models/exp2/ot_emd_13b_wrapper.pt \
  $COMMON_ARGS --name /workspace/results/ot-emd-full

echo "=== Computing Summary ==="
python3 -c "
import orjson, json
from pathlib import Path

results = {}
for run in ['kmeans-full', 'spectral-full', 'ot-emd-full']:
    scores_dir = Path(f'/workspace/results/{run}/scores/detection')
    if not scores_dir.exists():
        print(f'{run}: no scores found, skipping')
        continue
    rows = []
    for f in scores_dir.glob('*.txt'):
        rows.extend(orjson.loads(f.read_bytes()))

    total = len(rows)
    tp = sum(1 for r in rows if r['prediction'] and r['activating'])
    tn = sum(1 for r in rows if not r['prediction'] and not r['activating'])
    fp = sum(1 for r in rows if r['prediction'] and not r['activating'])
    fn = sum(1 for r in rows if not r['prediction'] and r['activating'])

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    bal_acc   = ((tp/(tp+fn) if (tp+fn)>0 else 0) + (tn/(tn+fp) if (tn+fp)>0 else 0)) / 2

    results[run] = dict(
        total=total,
        balanced_acc=round(bal_acc, 3),
        f1=round(f1, 3),
        precision=round(precision, 3),
        recall=round(recall, 3),
        tp=tp, tn=tn, fp=fp, fn=fn
    )

header = f\"{'Model':<20} {'Bal Acc':>8} {'F1':>8} {'Precision':>10} {'Recall':>8} {'TP':>6} {'TN':>6} {'FP':>6} {'FN':>6}\"
sep = '-' * len(header)
print(header)
print(sep)
for run, m in results.items():
    print(f\"{run:<20} {m['balanced_acc']:>8} {m['f1']:>8} {m['precision']:>10} {m['recall']:>8} {m['tp']:>6} {m['tn']:>6} {m['fp']:>6} {m['fn']:>6}\")

with open('/workspace/results/summary.json', 'w') as f:
    json.dump(results, f, indent=2)
print()
print('Saved to /workspace/results/summary.json')
" | tee /workspace/results/summary_table.txt

echo "=== All done! ==="
echo "Table: /workspace/results/summary_table.txt"
echo "JSON:  /workspace/results/summary.json"
