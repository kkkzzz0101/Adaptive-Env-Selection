#!/usr/bin/env python3
from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path

from src.scheduler import AdaptiveCurriculumScheduler, SchedulerConfig


def build_levels_and_calibration():
    # Higher accuracy => easier.
    levels = {
        "countdown": {1: 0.78, 2: 0.63, 3: 0.48, 4: 0.33, 5: 0.19},
        "zebra": {3: 0.72, 4: 0.57, 5: 0.41, 6: 0.28, 7: 0.16},
        "arc1d": {1: 0.69, 2: 0.52, 3: 0.36, 4: 0.22},
        "math": {1: 0.74, 2: 0.58, 3: 0.43, 4: 0.29, 5: 0.14},
    }
    calibration = {}
    for ds, lv_map in levels.items():
        for raw, acc in lv_map.items():
            calibration[(ds, raw)] = acc
    return levels, calibration


def build_samples(levels, per_level=30):
    samples = []
    for ds, lv_map in levels.items():
        for raw in lv_map:
            for i in range(per_level):
                samples.append(
                    {
                        "sample_id": f"{ds}_{raw}_{i}",
                        "dataset_id": ds,
                        "raw_difficulty": raw,
                    }
                )
    return samples


def expected_level_cluster(calibration, num_clusters):
    # Mirror scheduler split logic by global rank on accuracy descending.
    items = sorted(calibration.items(), key=lambda x: x[1], reverse=True)
    keys = [k for k, _ in items]
    n = len(keys)
    out = {}
    for cid in range(num_clusters):
        start = (cid * n) // num_clusters
        end = ((cid + 1) * n) // num_clusters
        for k in keys[start:end]:
            out[k] = cid
    return out


def run_eval(seed=42):
    rng = random.Random(seed)
    num_clusters = 4

    levels, calibration = build_levels_and_calibration()
    samples = build_samples(levels, per_level=30)

    cfg = SchedulerConfig(
        num_clusters=num_clusters,
        decay=0.95,
        ucb_beta=1.0,
        softmax_tau=0.2,
        prob_floor_eps=0.05,
        warmup_steps=20,
        rebucket_interval=10,
        active_window=80,
        min_obs_for_rebucket=4,
        migration_gamma=1.5,
        migration_consecutive=2,
        allow_only_neighbor_migration=True,
        allow_reverse_migration=False,
        seed=seed,
    )

    scheduler = AdaptiveCurriculumScheduler(samples=samples, config=cfg, calibration_map=calibration)

    # 1) Initial clustering quality check
    expected = expected_level_cluster(calibration, num_clusters)
    level_assignments = defaultdict(lambda: defaultdict(int))
    for sid, st in scheduler.samples.items():
        level_assignments[(st.dataset_id, st.raw_difficulty)][st.cluster_id] += 1

    level_report = {}
    correct_levels = 0
    for key, counts in sorted(level_assignments.items()):
        dominant = max(counts.items(), key=lambda x: x[1])[0]
        exp = expected[key]
        ok = dominant == exp
        if ok:
            correct_levels += 1
        level_report[f"{key[0]}:{key[1]}"] = {
            "dominant_cluster": dominant,
            "expected_cluster": exp,
            "cluster_counts": dict(sorted(counts.items())),
            "match": ok,
        }

    # 2) A tracking + UCB sampling (with controlled rewards)
    #    Harder clusters get slightly larger |A| to force non-uniform UCB.
    outlier_id = None
    # pick one sample currently in hardest cluster (3) for rebucketing test
    for sid, st in scheduler.samples.items():
        if st.cluster_id == num_clusters - 1:
            outlier_id = sid
            break

    prob_history = []
    migration_events = []
    for step in range(1, 121):
        batch_ids, _ = scheduler.sample_batch(32)
        abs_adv = []
        for sid in batch_ids:
            st = scheduler.samples[sid]
            base = 0.15 + 0.10 * st.cluster_id
            noise = rng.uniform(-0.03, 0.03)
            adv = max(0.0, base + noise)
            # 3) persistent outlier after warm-up to trigger re-bucketing
            if sid == outlier_id and step >= 30:
                adv = 1.2
            abs_adv.append(adv)

        scheduler.update_after_batch(batch_ids, abs_adv, global_step=step)
        mig = scheduler.maybe_rebucket(global_step=step)
        if mig:
            migration_events.extend([{**m, "step": step} for m in mig])

        if step % 10 == 0:
            stats = scheduler.get_cluster_stats()
            prob_history.append(
                {
                    "step": step,
                    "values": {str(k): round(v["value"], 4) for k, v in stats.items()},
                    "probs": {str(k): round(v["prob"], 4) for k, v in stats.items()},
                    "draws": {str(k): v["sample_count"] for k, v in stats.items()},
                }
            )

    final_stats = scheduler.get_cluster_stats()

    # checks
    probs = [final_stats[c]["prob"] for c in sorted(final_stats)]
    prob_sum_ok = abs(sum(probs) - 1.0) < 1e-6
    prob_non_uniform = (max(probs) - min(probs)) > 0.02

    # migration checks: neighbor-only and easier-only
    neighbor_only_ok = True
    easier_only_ok = True
    for m in migration_events:
        if abs(m["to_cluster"] - m["from_cluster"]) != 1:
            neighbor_only_ok = False
        if m["to_cluster"] >= m["from_cluster"]:
            easier_only_ok = False

    out = {
        "config": {
            "num_clusters": num_clusters,
            "samples": len(samples),
            "levels": len(level_report),
        },
        "initial_clustering": {
            "level_match": f"{correct_levels}/{len(level_report)}",
            "all_levels_match": correct_levels == len(level_report),
            "details": level_report,
        },
        "a_tracking_ucb": {
            "prob_sum_ok": prob_sum_ok,
            "prob_non_uniform": prob_non_uniform,
            "final_cluster_stats": final_stats,
            "history_every_10_steps": prob_history,
        },
        "rebucketing": {
            "migrations_total": len(migration_events),
            "neighbor_only_ok": neighbor_only_ok,
            "easier_only_ok": easier_only_ok,
            "events": migration_events[:20],
        },
    }
    return out


if __name__ == "__main__":
    report = run_eval(seed=42)
    out_path = Path("experiments/baselines/results_512/scheduler_light_eval_report.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print("[DONE] scheduler_light_eval")
    print(f"[OUT] {out_path}")
    print(json.dumps({
        "initial_all_levels_match": report["initial_clustering"]["all_levels_match"],
        "prob_sum_ok": report["a_tracking_ucb"]["prob_sum_ok"],
        "prob_non_uniform": report["a_tracking_ucb"]["prob_non_uniform"],
        "migrations_total": report["rebucketing"]["migrations_total"],
        "neighbor_only_ok": report["rebucketing"]["neighbor_only_ok"],
        "easier_only_ok": report["rebucketing"]["easier_only_ok"],
    }, ensure_ascii=False))
