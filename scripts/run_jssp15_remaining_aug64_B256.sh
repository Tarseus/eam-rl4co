#!/usr/bin/env bash
set -euo pipefail

cd /data1/gushengda/eam-rl4co
export PYTHONPATH=/data1/gushengda/eam-rl4co
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2}"

PY=/data1/gushengda/anaconda3/envs/rlco1/bin/python
STAMP=$(date +%Y%m%d-%H%M%S)
ROOT=logs/codex_remote/jssp15x15_remaining_aug64_B256_${STAMP}
mkdir -p "$ROOT"
echo "root=$ROOT"

for spec in \
  rl:downloads/jssp15x15/rl/checkpoint.ckpt \
  po:downloads/jssp15x15/po/checkpoint.ckpt \
  sll:downloads/jssp15x15/sll/checkpoint.ckpt; do
  method=${spec%%:*}
  ckpt=${spec#*:}
  out="$ROOT/$method"
  echo "=== $method $ckpt ==="
  "$PY" scripts/eval_jssp_benchmarks.py \
    --model-path "$ckpt" \
    --device cuda:0 \
    --B 256 \
    --aug-factor 64 \
    --aug-batch-size 4 \
    --aug-seed 20260504 \
    --sets \
    --ood-dir jssp15x15=data/jssp_bopo/validation_15x15 \
    --skip-prepare \
    --output-dir "$out"
done

"$PY" - <<'PY'
import csv
import json
import pathlib

roots = sorted(pathlib.Path("logs/codex_remote").glob("jssp15x15_remaining_aug64_B256_*"))
root = roots[-1]
rows = []
for d in sorted(p for p in root.iterdir() if p.is_dir()):
    s = json.loads((d / "summary.json").read_text())
    info = s["sets"]["jssp15x15"]
    csv_path = pathlib.Path(info["csv_path"])
    pred_rows = list(csv.DictReader(csv_path.open()))
    rows.append(
        {
            "method": d.name,
            "avg_makespan": sum(float(r["pred_makespan"]) for r in pred_rows) / len(pred_rows),
            "avg_gap": info["avg_gap"],
            "avg_time_sec": info["avg_time_sec"],
            "count": info["count"],
            "csv_path": info["csv_path"],
        }
    )
with (root / "summary_methods.csv").open("w", newline="") as f:
    w = csv.DictWriter(
        f,
        fieldnames=["method", "avg_makespan", "avg_gap", "avg_time_sec", "count", "csv_path"],
    )
    w.writeheader()
    w.writerows(rows)
print("summary_csv", root / "summary_methods.csv")
for r in rows:
    print(r)
PY
