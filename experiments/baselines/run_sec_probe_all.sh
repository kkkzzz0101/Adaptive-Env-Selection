#!/usr/bin/env bash
set -euo pipefail

ROOT='/root/adaptive env selection'
PYTHON=${PYTHON:-python}
MODEL_PATH=${MODEL_PATH:-/root/models/Qwen2.5-1.5B-Instruct}
N_PER_DIFFICULTY=${N_PER_DIFFICULTY:-20}
ROLLOUTS=${ROLLOUTS:-1}
TEMPERATURE=${TEMPERATURE:-0}
MAX_NEW_TOKENS_PUZZLE=${MAX_NEW_TOKENS_PUZZLE:-180}
MAX_NEW_TOKENS_MATH=${MAX_NEW_TOKENS_MATH:-220}
OUT_DIR=${OUT_DIR:-$ROOT/experiments/baselines/results_512/sec_probe_all_instruct_n${N_PER_DIFFICULTY}}

mkdir -p "$OUT_DIR"

run_probe() {
  local dataset="$1"
  local split="$2"
  local max_new="$3"
  echo "[RUN] dataset=${dataset} split=${split} n_per_difficulty=${N_PER_DIFFICULTY}"
  "$PYTHON" "$ROOT/scripts/sec_inference_probe.py" \
    --dataset "$dataset" \
    --split "$split" \
    --n-per-difficulty "$N_PER_DIFFICULTY" \
    --rollouts "$ROLLOUTS" \
    --max-new-tokens "$max_new" \
    --temperature "$TEMPERATURE" \
    --model-path "$MODEL_PATH" \
    --out-dir "$OUT_DIR"
}

run_probe zebra test "$MAX_NEW_TOKENS_PUZZLE"
run_probe arc test "$MAX_NEW_TOKENS_PUZZLE"
run_probe countdown test "$MAX_NEW_TOKENS_PUZZLE"
run_probe math train "$MAX_NEW_TOKENS_MATH"

"$PYTHON" - <<'PY'
from pathlib import Path
import pandas as pd

out_dir = Path('/root/adaptive env selection/experiments/baselines/results_512')
# Find the newest sec_probe_all_instruct_n* directory
cands = sorted([p for p in out_dir.glob('sec_probe_all_instruct_n*') if p.is_dir()], key=lambda x: x.stat().st_mtime, reverse=True)
if not cands:
    raise SystemExit('No result directory found.')
base = cands[0]

pairs = [
    ('zebra', 'test'),
    ('arc', 'test'),
    ('countdown', 'test'),
    ('math', 'train'),
]

frames = []
for ds, split in pairs:
    p = base / f'{ds}_{split}_summary.csv'
    if not p.exists():
        print(f'[WARN] missing {p}')
        continue
    df = pd.read_csv(p)
    df.insert(0, 'dataset', ds)
    df.insert(1, 'split', split)
    frames.append(df)

if not frames:
    raise SystemExit('No summary csv found.')

all_df = pd.concat(frames, ignore_index=True)
out_csv = base / 'all_datasets_summary.csv'
out_md = base / 'all_datasets_summary.md'
all_df.to_csv(out_csv, index=False)

with out_md.open('w', encoding='utf-8') as f:
    f.write('# SEC Probe Summary (All Datasets)\n\n')
    for ds in ['zebra', 'arc', 'countdown', 'math']:
        sub = all_df[all_df['dataset'] == ds]
        if sub.empty:
            continue
        f.write(f'## {ds}\n\n')
        f.write(sub.to_markdown(index=False))
        f.write('\n\n')

print(f'[OK] merged summary -> {out_csv}')
print(f'[OK] markdown report -> {out_md}')
print(all_df.to_string(index=False))
PY

echo "[DONE] all datasets probe finished. out_dir=$OUT_DIR"
