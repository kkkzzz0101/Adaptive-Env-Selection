#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd

sys.path.insert(0, '/root/adaptive env selection')
from src.scheduler import AdaptiveCurriculumScheduler, SchedulerConfig  # noqa: E402


def _parse_dataset_and_raw(data_source: str) -> Tuple[str, Any] | None:
    s = str(data_source)
    if s.startswith('countdown_diff_'):
        return 'countdown', int(s.split('_')[-1])
    if s.startswith('zebra_houses_'):
        return 'zebra', int(s.split('_')[-1])
    if s.startswith('math_train_level_'):
        return 'math_train', int(s.split('_')[-1])
    return None


def _build_default_calibration() -> Dict[Tuple[str, Any], float]:
    # Higher accuracy => easier.
    rows = [
        ('countdown', 1, 0.70),
        ('countdown', 2, 0.50),
        ('countdown', 3, 0.32),
        ('countdown', 4, 0.18),
        ('zebra', 3, 0.65),
        ('zebra', 4, 0.45),
        ('zebra', 5, 0.28),
        ('zebra', 6, 0.15),
        ('math_train', 1, 0.72),
        ('math_train', 2, 0.55),
        ('math_train', 3, 0.38),
        ('math_train', 4, 0.24),
        ('math_train', 5, 0.12),
    ]
    return {(d, lvl): acc for d, lvl, acc in rows}


def _load_small_samples(parquet_path: Path, max_per_source: int, seed: int) -> List[Dict[str, Any]]:
    df = pd.read_parquet(parquet_path)

    records: List[Dict[str, Any]] = []
    rng = random.Random(seed)

    for ds, group in df.groupby('data_source'):
        parsed = _parse_dataset_and_raw(str(ds))
        if parsed is None:
            continue
        dataset_id, raw = parsed

        idxs = list(group.index)
        rng.shuffle(idxs)
        keep = idxs[:max_per_source]
        for i in keep:
            records.append(
                {
                    'sample_id': f'sample_{i}',
                    'dataset_id': dataset_id,
                    'raw_difficulty': raw,
                }
            )

    if not records:
        raise RuntimeError('No usable samples loaded for countdown/zebra/math_train')

    rng.shuffle(records)
    return records


def _synthetic_abs_adv(sample: Dict[str, Any], step: int, rng: random.Random) -> float:
    # Harder raw difficulty gets larger expected |A|, plus small per-step drift.
    raw = int(sample['raw_difficulty'])
    base = 0.10 + 0.08 * raw
    drift = 0.03 * (1.0 if step > 120 else 0.0)
    noise = rng.uniform(-0.05, 0.05)
    return max(0.0, base + drift + noise)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--train-parquet', default='/root/adaptive env selection/experiments/baselines/data_formal/mixed/train.parquet')
    ap.add_argument('--num-clusters', type=int, default=3)
    ap.add_argument('--steps', type=int, default=180)
    ap.add_argument('--batch-size', type=int, default=32)
    ap.add_argument('--max-per-source', type=int, default=20)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--out-json', default='/root/adaptive env selection/experiments/baselines/results_1024/scheduler_smoke_report.json')
    args = ap.parse_args()

    samples = _load_small_samples(Path(args.train_parquet), args.max_per_source, args.seed)
    calibration = _build_default_calibration()

    cfg = SchedulerConfig(
        num_clusters=args.num_clusters,
        decay=0.95,
        ucb_beta=1.0,
        softmax_tau=0.2,
        prob_floor_eps=0.05,
        warmup_steps=60,
        rebucket_interval=20,
        active_window=80,
        min_obs_for_rebucket=5,
        migration_gamma=2.0,
        migration_consecutive=2,
        allow_only_neighbor_migration=True,
        allow_reverse_migration=False,
        seed=args.seed,
    )

    scheduler = AdaptiveCurriculumScheduler(samples=samples, config=cfg, calibration_map=calibration)

    rng = random.Random(args.seed + 7)
    migration_events: List[Dict[str, Any]] = []

    sample_lookup = {s['sample_id']: s for s in samples}

    for step in range(1, args.steps + 1):
        batch_ids, batch_clusters = scheduler.sample_batch(args.batch_size)
        abs_adv = [_synthetic_abs_adv(sample_lookup[sid], step, rng) for sid in batch_ids]
        scheduler.update_after_batch(batch_ids, abs_adv, global_step=step)

        mig = scheduler.maybe_rebucket(global_step=step)
        if mig:
            migration_events.extend(mig)

        if step % 30 == 0 or step == args.steps:
            stats = scheduler.get_cluster_stats()
            compact = {
                str(cid): {
                    'size': v['size'],
                    'value': round(v['value'], 4),
                    'prob': round(v['prob'], 4),
                    'draws': v['sample_count'],
                    'mig_in': v['migration_in'],
                    'mig_out': v['migration_out'],
                }
                for cid, v in stats.items()
            }
            print(f'[step {step}] {json.dumps(compact, ensure_ascii=False)}')

    final_cluster = scheduler.get_cluster_stats()
    all_samples = scheduler.get_sample_stats()
    migrated_count = sum(1 for s in all_samples.values() if s['migration_history'])

    report = {
        'config': {
            'num_clusters': args.num_clusters,
            'steps': args.steps,
            'batch_size': args.batch_size,
            'sample_pool_size': len(samples),
        },
        'total_draws': sum(v['sample_count'] for v in final_cluster.values()),
        'migrations_total': len(migration_events),
        'migrated_samples': migrated_count,
        'cluster_stats': final_cluster,
        'sample_preview': dict(list(all_samples.items())[:10]),
    }

    out = Path(args.out_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')

    print('[DONE] scheduler smoke finished')
    print(f'[OUT] {out}')
    print(
        json.dumps(
            {
                'migrations_total': report['migrations_total'],
                'migrated_samples': report['migrated_samples'],
                'total_draws': report['total_draws'],
            },
            ensure_ascii=False,
        )
    )


if __name__ == '__main__':
    main()
