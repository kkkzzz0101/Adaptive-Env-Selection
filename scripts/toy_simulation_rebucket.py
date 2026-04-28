#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
import sys
from typing import Dict, List

import matplotlib.pyplot as plt

# Allow running this script directly from repo root without package install.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.scheduler import AdaptiveCurriculumScheduler, SchedulerConfig


def build_toy_scheduler(num_per_cluster: int = 50, seed: int = 42) -> AdaptiveCurriculumScheduler:
    cfg = SchedulerConfig(seed=seed)
    samples: List[Dict] = []
    for cid in range(cfg.num_clusters):
        for i in range(num_per_cluster):
            samples.append(
                {
                    "sample_id": f"c{cid}_s{i}",
                    "dataset_id": "toy",
                    "raw_difficulty": cid,
                    "cluster_id": cid,
                }
            )
    return AdaptiveCurriculumScheduler(samples=samples, config=cfg)


def set_cluster_distribution(
    scheduler: AdaptiveCurriculumScheduler,
    cluster_id: int,
    mean: float,
    std: float,
    step: int,
    rng: random.Random,
    *,
    overrides: Dict[str, float] | None = None,
) -> None:
    members = sorted(scheduler.clusters[cluster_id].member_ids)
    for sid in members:
        state = scheduler.samples[sid]
        if overrides and sid in overrides:
            val = overrides[sid]
        else:
            val = max(0.0, rng.gauss(mean, std))
        state.s = float(val)
        state.obs_count = max(state.obs_count, scheduler.cfg.min_obs_for_rebucket)
        state.last_update_step = step


def candidate_count(scheduler: AdaptiveCurriculumScheduler, cluster_id: int) -> int:
    c = 0
    for sid in scheduler.clusters[cluster_id].member_ids:
        st = scheduler.samples[sid]
        if st.upward_streak <= 0:
            continue
        if st.last_update_step < 0:
            continue
        if (scheduler.global_step - st.last_update_step) > scheduler.cfg.active_window:
            continue
        c += 1
    return c


def print_refresh_summary(
    scenario_name: str,
    refresh_idx: int,
    step: int,
    scheduler: AdaptiveCurriculumScheduler,
    migrations: List[Dict],
) -> None:
    stats = scheduler.get_cluster_stats()
    print(f"\n[{scenario_name}] refresh={refresh_idx} step={step} migrated_count={len(migrations)}")
    for cid in range(scheduler.cfg.num_clusters):
        row = stats[cid]
        cand = candidate_count(scheduler, cid)
        print(
            f"  cluster={row['cluster_label']} size={row['size']} active={row['active_size']} "
            f"mean={row['active_mean']:.4f} std={row['active_std']:.4f} candidates={cand}"
        )

    for m in migrations:
        event = {
            "step": m["step"],
            "sample_id": m["sample_id"],
            "old_cluster": scheduler.cluster_labels[m["from_cluster"]],
            "new_cluster": scheduler.cluster_labels[m["to_cluster"]],
            "s_i": round(m["s_i"], 4),
            "cluster_mean": round(m["mu"], 4),
            "cluster_std": round(m["sigma"], 4),
            "delta": round(m["delta"], 4),
            "obs_count": m["obs_count"],
            "consecutive_trigger_count": m["consecutive_trigger_count"],
        }
        print("  migration_event=" + json.dumps(event, ensure_ascii=False))


def run_scenario_a(rng: random.Random) -> Dict:
    scheduler = build_toy_scheduler(seed=11)
    cfg = scheduler.cfg

    medium_means = [0.40, 0.45, 0.50]
    std = 0.05
    tracked_id = "c1_s0"

    history = {"name": "A_normal_drifting", "steps": [], "medium_mean": [], "tracked": [], "migration_steps": []}

    start_step = cfg.warmup_steps
    for idx, mean in enumerate(medium_means, start=1):
        step = start_step + (idx - 1) * cfg.rebucket_interval

        set_cluster_distribution(scheduler, 0, mean=0.20, std=std, step=step, rng=rng)
        set_cluster_distribution(scheduler, 1, mean=mean, std=std, step=step, rng=rng)
        set_cluster_distribution(scheduler, 2, mean=0.70, std=std, step=step, rng=rng)

        scheduler._refresh_cluster_stats()
        scheduler._refresh_ucb_and_probs()
        migrations = scheduler.maybe_rebucket(global_step=step)
        print_refresh_summary("Scenario A", idx, step, scheduler, migrations)

        history["steps"].append(step)
        history["medium_mean"].append(scheduler.get_cluster_stats()[1]["active_mean"])
        history["tracked"].append(scheduler.samples[tracked_id].s)
        history["migration_steps"].extend([m["step"] for m in migrations])

    return history


def run_scenario_b(rng: random.Random) -> Dict:
    scheduler = build_toy_scheduler(seed=22)
    cfg = scheduler.cfg

    medium_means = [0.40, 0.45, 0.50]
    outlier_values = [0.55, 0.62, 0.70]
    std = 0.05
    outlier_id = "c1_s0"

    history = {"name": "B_persistent_outlier", "steps": [], "medium_mean": [], "tracked": [], "migration_steps": []}

    start_step = cfg.warmup_steps
    for idx, (mean, outlier_val) in enumerate(zip(medium_means, outlier_values), start=1):
        step = start_step + (idx - 1) * cfg.rebucket_interval

        set_cluster_distribution(scheduler, 0, mean=0.20, std=std, step=step, rng=rng)
        set_cluster_distribution(scheduler, 1, mean=mean, std=std, step=step, rng=rng, overrides={outlier_id: outlier_val})
        set_cluster_distribution(scheduler, 2, mean=0.70, std=std, step=step, rng=rng)

        scheduler._refresh_cluster_stats()
        scheduler._refresh_ucb_and_probs()
        migrations = scheduler.maybe_rebucket(global_step=step)
        print_refresh_summary("Scenario B", idx, step, scheduler, migrations)

        history["steps"].append(step)
        history["medium_mean"].append(scheduler.get_cluster_stats()[1]["active_mean"])
        history["tracked"].append(outlier_val)
        history["migration_steps"].extend([m["step"] for m in migrations if m["sample_id"] == outlier_id])

    return history


def run_scenario_c(rng: random.Random) -> Dict:
    scheduler = build_toy_scheduler(seed=33)
    cfg = scheduler.cfg

    medium_means = [0.40, 0.45, 0.50]
    spike_values = [0.43, 0.70, 0.46]
    std = 0.05
    spike_id = "c1_s0"

    history = {"name": "C_single_spike", "steps": [], "medium_mean": [], "tracked": [], "migration_steps": []}

    start_step = cfg.warmup_steps
    for idx, (mean, spike_val) in enumerate(zip(medium_means, spike_values), start=1):
        step = start_step + (idx - 1) * cfg.rebucket_interval

        set_cluster_distribution(scheduler, 0, mean=0.20, std=std, step=step, rng=rng)
        set_cluster_distribution(scheduler, 1, mean=mean, std=std, step=step, rng=rng, overrides={spike_id: spike_val})
        set_cluster_distribution(scheduler, 2, mean=0.70, std=std, step=step, rng=rng)

        scheduler._refresh_cluster_stats()
        scheduler._refresh_ucb_and_probs()
        migrations = scheduler.maybe_rebucket(global_step=step)
        print_refresh_summary("Scenario C", idx, step, scheduler, migrations)

        history["steps"].append(step)
        history["medium_mean"].append(scheduler.get_cluster_stats()[1]["active_mean"])
        history["tracked"].append(spike_val)
        history["migration_steps"].extend([m["step"] for m in migrations if m["sample_id"] == spike_id])

    return history


def plot_histories(histories: List[Dict], output_path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(16, 4), sharey=False)

    for ax, hist in zip(axes, histories):
        steps = hist["steps"]
        ax.plot(steps, hist["medium_mean"], marker="o", label="cluster mean (medium)")
        ax.plot(steps, hist["tracked"], marker="s", label="tracked sample state")

        for ms in hist["migration_steps"]:
            ax.axvline(ms, color="red", linestyle="--", alpha=0.7)
            ax.text(ms, max(hist["tracked"]), "migration", color="red", fontsize=8)

        ax.set_title(hist["name"])
        ax.set_xlabel("refresh step")
        ax.set_ylabel("state")
        ax.grid(alpha=0.2)
        ax.legend(fontsize=8)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Toy simulation for scheduler re-bucketing behavior")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--artifacts-dir", type=str, default="artifacts")
    args = parser.parse_args()

    rng = random.Random(args.seed)

    hist_a = run_scenario_a(rng)
    hist_b = run_scenario_b(rng)
    hist_c = run_scenario_c(rng)

    artifacts_dir = Path(args.artifacts_dir)
    plot_path = artifacts_dir / "toy_rebucket_simulation.png"
    plot_histories([hist_a, hist_b, hist_c], plot_path)

    summary_path = artifacts_dir / "toy_rebucket_summary.json"
    summary_path.write_text(
        json.dumps({"scenarios": [hist_a, hist_b, hist_c]}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("\n[DONE] toy simulation finished")
    print(f"[ARTIFACT] plot={plot_path}")
    print(f"[ARTIFACT] summary={summary_path}")


if __name__ == "__main__":
    main()
