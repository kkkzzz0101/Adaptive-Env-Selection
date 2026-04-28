from __future__ import annotations

import math

import pytest

from src.scheduler import AdaptiveCurriculumScheduler, SchedulerConfig


def _make_samples(counts_by_cluster: dict[int, int]) -> list[dict]:
    samples = []
    for cid, n in counts_by_cluster.items():
        for i in range(n):
            samples.append(
                {
                    "sample_id": f"c{cid}_s{i}",
                    "dataset_id": "toy",
                    "raw_difficulty": cid,
                    "cluster_id": cid,
                }
            )
    return samples


def _build_scheduler(counts_by_cluster: dict[int, int], **cfg_kwargs) -> AdaptiveCurriculumScheduler:
    cfg = SchedulerConfig(**cfg_kwargs)
    return AdaptiveCurriculumScheduler(samples=_make_samples(counts_by_cluster), config=cfg)


def _softmax(scores: list[float], tau: float) -> list[float]:
    m = max(scores)
    logits = [math.exp((x - m) / tau) for x in scores]
    z = sum(logits)
    return [x / z for x in logits]


def test_probability_normalization() -> None:
    scheduler = _build_scheduler({0: 20, 1: 20, 2: 20}, softmax_tau=0.5, prob_floor_eps=0.05)

    scheduler.clusters[0].value = 0.1
    scheduler.clusters[1].value = 0.6
    scheduler.clusters[2].value = 1.1
    scheduler.clusters[0].sample_count = 50
    scheduler.clusters[1].sample_count = 50
    scheduler.clusters[2].sample_count = 50
    scheduler.total_draws = 150
    scheduler._refresh_ucb_and_probs()

    stats = scheduler.get_cluster_stats()
    probs = [stats[cid]["prob"] for cid in range(3)]
    ucb_scores = [stats[cid]["ucb_score"] for cid in range(3)]
    raw_softmax = _softmax(ucb_scores, tau=scheduler.cfg.softmax_tau)

    assert all(0.0 <= p <= 1.0 for p in probs)
    assert sum(probs) == pytest.approx(1.0, abs=1e-9)
    assert raw_softmax[2] > raw_softmax[1] > raw_softmax[0]


def test_probability_floor() -> None:
    scheduler = _build_scheduler({0: 10, 1: 10, 2: 10}, softmax_tau=0.2, prob_floor_eps=0.06)

    scheduler.clusters[0].value = 20.0
    scheduler.clusters[1].value = 20.0
    scheduler.clusters[2].value = -100.0
    scheduler.total_draws = 100
    scheduler._refresh_ucb_and_probs()

    low_prob = scheduler.get_cluster_stats()[2]["prob"]
    theoretical_min_floor = scheduler.cfg.prob_floor_eps / scheduler.cfg.num_clusters

    assert low_prob > 0.0
    assert low_prob >= theoretical_min_floor - 1e-12


def test_with_replacement_sampling() -> None:
    scheduler = _build_scheduler({0: 5, 1: 0, 2: 0}, prob_floor_eps=0.0)
    original_members = set(scheduler.clusters[0].member_ids)

    sampled_ids, sampled_clusters = scheduler.sample_batch(batch_size=200)

    assert len(set(sampled_ids)) < len(sampled_ids)
    assert set(sampled_ids).issubset(original_members)
    assert set(sampled_clusters) == {0}
    assert len(scheduler.clusters[0].member_ids) == 5
    assert scheduler.clusters[0].member_ids == original_members


def test_decayed_state_update() -> None:
    scheduler = _build_scheduler({1: 1}, decay=0.9)
    sid = next(iter(scheduler.samples.keys()))

    scheduler.update_after_batch([sid], [1.0], global_step=1)
    assert scheduler.samples[sid].s == pytest.approx(0.1, abs=1e-9)

    scheduler.update_after_batch([sid], [1.0], global_step=2)
    assert scheduler.samples[sid].s == pytest.approx(0.19, abs=1e-9)

    scheduler.update_after_batch([sid], [0.0], global_step=3)
    assert scheduler.samples[sid].s == pytest.approx(0.171, abs=1e-9)


def test_warmup_no_migration() -> None:
    scheduler = _build_scheduler(
        {1: 12},
        warmup_steps=200,
        rebucket_interval=1,
        min_obs_for_rebucket=1,
        migration_consecutive=1,
        migration_gamma=1.0,
    )

    outlier_id = "c1_s0"
    for sid, state in scheduler.samples.items():
        state.obs_count = 20
        state.last_update_step = 100
        state.s = 1.0 if sid == outlier_id else 0.1

    migrations = scheduler.maybe_rebucket(global_step=100)

    assert migrations == []
    assert scheduler.samples[outlier_id].cluster_id == 1
    assert scheduler.samples[outlier_id].upward_streak == 0


def test_min_obs_guard() -> None:
    scheduler = _build_scheduler(
        {1: 12},
        warmup_steps=0,
        rebucket_interval=1,
        min_obs_for_rebucket=10,
        migration_consecutive=1,
        migration_gamma=1.0,
    )

    outlier_id = "c1_s0"
    for sid, state in scheduler.samples.items():
        state.last_update_step = 50
        state.s = 1.0 if sid == outlier_id else 0.1
        state.obs_count = 3 if sid == outlier_id else 20

    migrations = scheduler.maybe_rebucket(global_step=50)

    assert migrations == []
    assert scheduler.samples[outlier_id].cluster_id == 1
    assert scheduler.samples[outlier_id].upward_streak == 0


def test_neighbor_only_migration() -> None:
    hard_scheduler = _build_scheduler(
        {0: 10, 1: 10, 2: 20},
        warmup_steps=0,
        rebucket_interval=1,
        min_obs_for_rebucket=1,
        migration_consecutive=2,
        migration_gamma=1.0,
        allow_only_neighbor_migration=True,
        allow_reverse_migration=False,
    )

    hard_outlier_id = "c2_s0"
    for sid, state in hard_scheduler.samples.items():
        state.last_update_step = 10
        state.obs_count = 20
        state.s = 1.0 if sid == hard_outlier_id else 0.1

    assert hard_scheduler.maybe_rebucket(global_step=10) == []
    hard_migrations = hard_scheduler.maybe_rebucket(global_step=11)
    assert len(hard_migrations) == 1
    assert hard_migrations[0]["sample_id"] == hard_outlier_id
    assert hard_migrations[0]["from_cluster"] == 2
    assert hard_migrations[0]["to_cluster"] == 1
    assert hard_scheduler.samples[hard_outlier_id].cluster_id == 1

    medium_scheduler = _build_scheduler(
        {0: 10, 1: 20, 2: 10},
        warmup_steps=0,
        rebucket_interval=1,
        min_obs_for_rebucket=1,
        migration_consecutive=2,
        migration_gamma=1.0,
        allow_only_neighbor_migration=True,
        allow_reverse_migration=False,
    )

    medium_outlier_id = "c1_s0"
    for sid, state in medium_scheduler.samples.items():
        state.last_update_step = 10
        state.obs_count = 20
        state.s = 1.0 if sid == medium_outlier_id else 0.1

    assert medium_scheduler.maybe_rebucket(global_step=10) == []
    medium_migrations = medium_scheduler.maybe_rebucket(global_step=11)
    assert len(medium_migrations) == 1
    assert medium_migrations[0]["sample_id"] == medium_outlier_id
    assert medium_migrations[0]["from_cluster"] == 1
    assert medium_migrations[0]["to_cluster"] == 0
    assert medium_scheduler.samples[medium_outlier_id].cluster_id == 0
