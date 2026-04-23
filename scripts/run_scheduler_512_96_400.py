#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd

from src.scheduler import AdaptiveCurriculumScheduler, SchedulerConfig


SOURCE_PATTERNS = [
    (re.compile(r"^math_train_level_(\d+)$"), "math", "level"),
    (re.compile(r"^countdown_diff_(\d+)$"), "countdown", "diff"),
    (re.compile(r"^zebra_diff_(\d+)$"), "zebra", "diff"),
    (re.compile(r"^arc1d_diff_(\d+)$"), "arc1d", "diff"),
]


def parse_source(data_source: str) -> Tuple[str, int]:
    text = str(data_source)
    for pattern, dataset_id, _ in SOURCE_PATTERNS:
        m = pattern.match(text)
        if m:
            return dataset_id, int(m.group(1))

    # SEC-style sources without explicit bucket id.
    if text.startswith('countdown_'):
        return 'countdown', 1
    if text.startswith('zebra_'):
        return 'zebra', 1
    if text.startswith('arc_'):
        return 'arc1d', 1
    if text == 'math_train' or text == 'math500_test':
        return 'math', 1

    return "unknown", 1


def build_calibration_map(df: pd.DataFrame) -> Dict[Tuple[str, int], float]:
    grouped: Dict[str, List[int]] = defaultdict(list)
    for src in df["data_source"].astype(str).unique().tolist():
        ds, raw = parse_source(src)
        grouped[ds].append(raw)

    calibration: Dict[Tuple[str, int], float] = {}
    for ds, raws in grouped.items():
        uniq = sorted(set(raws))
        if len(uniq) == 1:
            calibration[(ds, uniq[0])] = 0.5
            continue
        for i, raw in enumerate(uniq):
            calibration[(ds, raw)] = 1.0 - (i / (len(uniq) - 1))
    return calibration


def build_samples(df: pd.DataFrame, max_prompt_len: int) -> List[Dict[str, Any]]:
    samples: List[Dict[str, Any]] = []
    for idx, row in df.reset_index(drop=True).iterrows():
        ds, raw = parse_source(row["data_source"])
        prompt = str(row.get("prompt", ""))
        samples.append(
            {
                "sample_id": f"s{idx}",
                "dataset_id": ds,
                "raw_difficulty": int(raw),
                "prompt_len": min(len(prompt), max_prompt_len),
                "data_source": str(row["data_source"]),
            }
        )
    return samples


def sample_one_third(df: pd.DataFrame, seed: int) -> pd.DataFrame:
    rng = random.Random(seed)
    keep_index: List[int] = []
    for _, group in df.groupby("data_source"):
        idxs = list(group.index)
        rng.shuffle(idxs)
        keep_n = max(1, len(idxs) // 3)
        keep_index.extend(idxs[:keep_n])
    subset = df.loc[sorted(keep_index)].reset_index(drop=True)
    return subset


def synthetic_abs_adv(sample: Dict[str, Any], step: int, rng: random.Random, cluster_id: int) -> float:
    len_term = min(1.0, float(sample.get("prompt_len", 0)) / 512.0)
    base = 0.08 + 0.07 * cluster_id + 0.03 * len_term
    drift = 0.02 if step > 200 else 0.0
    noise = rng.uniform(-0.025, 0.025)
    return max(0.0, base + drift + noise)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-parquet", default="experiments/baselines/data_sec4/mixed/train.parquet")
    ap.add_argument("--out-dir", default="experiments/baselines/results_512")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--num-clusters", type=int, default=4)
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--batch-size", type=int, default=24)
    ap.add_argument("--max-prompt-length", type=int, default=512)
    ap.add_argument("--max-response-length", type=int, default=96)
    args = ap.parse_args()

    full_df = pd.read_parquet(args.train_parquet)
    sub_df = sample_one_third(full_df, seed=args.seed)

    calibration = build_calibration_map(sub_df)
    samples = build_samples(sub_df, max_prompt_len=args.max_prompt_length)
    sample_lookup = {s["sample_id"]: s for s in samples}

    cfg = SchedulerConfig(
        num_clusters=args.num_clusters,
        decay=0.95,
        ucb_beta=1.0,
        softmax_tau=0.2,
        prob_floor_eps=0.05,
        warmup_steps=80,
        rebucket_interval=20,
        active_window=120,
        min_obs_for_rebucket=6,
        migration_gamma=2.0,
        migration_consecutive=2,
        allow_only_neighbor_migration=True,
        allow_reverse_migration=False,
        seed=args.seed,
    )

    scheduler = AdaptiveCurriculumScheduler(samples=samples, config=cfg, calibration_map=calibration)

    forced_outlier = None
    for sid, st in scheduler.samples.items():
        if st.cluster_id == (args.num_clusters - 1):
            forced_outlier = sid
            break

    rng = random.Random(args.seed + 11)
    prob_trace: List[Dict[str, Any]] = []
    migrations: List[Dict[str, Any]] = []

    init_stats = scheduler.get_cluster_stats()
    init_dist = {str(cid): v["size"] for cid, v in init_stats.items()}

    for step in range(1, args.steps + 1):
        batch_ids, batch_clusters = scheduler.sample_batch(args.batch_size)
        abs_adv: List[float] = []

        for sid, cid in zip(batch_ids, batch_clusters):
            val = synthetic_abs_adv(sample_lookup[sid], step, rng, cid)
            if forced_outlier is not None and sid == forced_outlier and step >= 140:
                val = 1.1
            abs_adv.append(val)

        scheduler.update_after_batch(batch_ids, abs_advantages=abs_adv, global_step=step)
        mig = scheduler.maybe_rebucket(global_step=step)
        if mig:
            for event in mig:
                e = dict(event)
                e["step"] = step
                migrations.append(e)

        if step % 20 == 0 or step == args.steps:
            stats = scheduler.get_cluster_stats()
            prob_trace.append(
                {
                    "step": step,
                    "probs": {str(k): round(v["prob"], 4) for k, v in stats.items()},
                    "values": {str(k): round(v["value"], 4) for k, v in stats.items()},
                    "draws": {str(k): int(v["sample_count"]) for k, v in stats.items()},
                    "sizes": {str(k): int(v["size"]) for k, v in stats.items()},
                }
            )

    final_stats = scheduler.get_cluster_stats()
    probs = [final_stats[c]["prob"] for c in sorted(final_stats)]

    report = {
        "run_name": "scheduler_only_sec4_1over3_512_96_400",
        "runtime_config": {
            "max_prompt_length": args.max_prompt_length,
            "max_response_length": args.max_response_length,
            "steps": args.steps,
            "batch_size": args.batch_size,
            "num_clusters": args.num_clusters,
        },
        "dataset": {
            "train_full_rows": int(len(full_df)),
            "train_subset_rows": int(len(sub_df)),
            "train_subset_ratio_real": float(len(sub_df) / max(1, len(full_df))),
            "data_source_counts_subset": sub_df["data_source"].value_counts().to_dict(),
        },
        "initial_grouping": {
            "cluster_sizes": init_dist,
        },
        "ucb_sampling": {
            "prob_sum": float(sum(probs)),
            "prob_trace": prob_trace,
            "final_cluster_stats": final_stats,
        },
        "rebucketing": {
            "migrations_total": len(migrations),
            "migrations_preview": migrations[:30],
            "neighbor_only_ok": all(abs(m["to_cluster"] - m["from_cluster"]) == 1 for m in migrations),
            "easier_only_ok": all(m["to_cluster"] < m["from_cluster"] for m in migrations),
        },
    }

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "scheduler_512_96_400_report.json"
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print("[DONE] scheduler 512/96 400-step run completed")
    print(f"[OUT] {out_path}")
    print(
        json.dumps(
            {
                "subset_rows": report["dataset"]["train_subset_rows"],
                "prob_sum": round(report["ucb_sampling"]["prob_sum"], 6),
                "migrations_total": report["rebucketing"]["migrations_total"],
                "neighbor_only_ok": report["rebucketing"]["neighbor_only_ok"],
                "easier_only_ok": report["rebucketing"]["easier_only_ok"],
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
