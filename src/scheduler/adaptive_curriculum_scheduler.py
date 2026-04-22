from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple
import math
import random


@dataclass
class SchedulerConfig:
    num_clusters: int = 3
    decay: float = 0.95
    ucb_beta: float = 1.0
    softmax_tau: float = 0.2
    prob_floor_eps: float = 0.05
    warmup_steps: int = 200
    rebucket_interval: int = 50
    active_window: int = 200
    min_obs_for_rebucket: int = 10
    migration_gamma: float = 2.0
    migration_consecutive: int = 3
    allow_only_neighbor_migration: bool = True
    allow_reverse_migration: bool = False
    seed: int = 42

    def __post_init__(self) -> None:
        if self.num_clusters < 2:
            raise ValueError("num_clusters must be >= 2")
        if not (0.0 < self.decay < 1.0):
            raise ValueError("decay must be in (0, 1)")
        if self.softmax_tau <= 0:
            raise ValueError("softmax_tau must be > 0")
        if not (0.0 <= self.prob_floor_eps < 1.0):
            raise ValueError("prob_floor_eps must be in [0, 1)")
        if self.migration_consecutive < 1:
            raise ValueError("migration_consecutive must be >= 1")


@dataclass
class SampleState:
    sample_id: str
    dataset_id: str
    raw_difficulty: Any
    cluster_id: int
    initial_cluster_id: int
    s: float = 0.0
    obs_count: int = 0
    last_update_step: int = -1
    upward_streak: int = 0
    downward_streak: int = 0
    migration_history: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class ClusterState:
    cluster_id: int
    member_ids: set[str] = field(default_factory=set)
    sample_count: int = 0
    value: float = 0.0
    active_count: int = 0
    active_mean: float = 0.0
    active_std: float = 0.0
    ucb_score: float = 0.0
    prob: float = 0.0
    migration_in: int = 0
    migration_out: int = 0


class AdaptiveCurriculumScheduler:
    """Difficulty-cluster scheduler with UCB sampling and low-frequency re-bucketing.

    Cluster index convention: 0 is easiest, num_clusters-1 is hardest.
    """

    def __init__(
        self,
        samples: Sequence[Mapping[str, Any]],
        config: Optional[SchedulerConfig] = None,
        *,
        calibration_map: Optional[Mapping[Tuple[str, Any], float]] = None,
    ) -> None:
        if not samples:
            raise ValueError("samples cannot be empty")

        self.cfg = config or SchedulerConfig()
        self.rng = random.Random(self.cfg.seed)
        self.global_step = 0
        self.total_draws = 0
        self.last_rebucket_step = -1

        self.cluster_labels = self._build_cluster_labels(self.cfg.num_clusters)
        self.samples: Dict[str, SampleState] = {}
        self.clusters: Dict[int, ClusterState] = {
            cid: ClusterState(cluster_id=cid, prob=1.0 / self.cfg.num_clusters)
            for cid in range(self.cfg.num_clusters)
        }

        initial_assignments = self._resolve_initial_clusters(samples, calibration_map)

        for item in samples:
            sid = str(item["sample_id"])
            if sid in self.samples:
                raise ValueError(f"duplicate sample_id detected: {sid}")

            dataset_id = str(item.get("dataset_id", "unknown"))
            raw_difficulty = item.get("raw_difficulty")
            cluster_id = int(initial_assignments[sid])

            state = SampleState(
                sample_id=sid,
                dataset_id=dataset_id,
                raw_difficulty=raw_difficulty,
                cluster_id=cluster_id,
                initial_cluster_id=cluster_id,
            )
            self.samples[sid] = state
            self.clusters[cluster_id].member_ids.add(sid)

        self._refresh_cluster_stats()
        self._refresh_ucb_and_probs()

    @staticmethod
    def _build_cluster_labels(num_clusters: int) -> List[str]:
        if num_clusters == 3:
            return ["easy", "medium", "hard"]
        return [f"cluster_{i}" for i in range(num_clusters)]

    def _resolve_initial_clusters(
        self,
        samples: Sequence[Mapping[str, Any]],
        calibration_map: Optional[Mapping[Tuple[str, Any], float]],
    ) -> Dict[str, int]:
        # Priority 1: explicit per-sample cluster_id / difficulty_band
        explicit: Dict[str, int] = {}
        for item in samples:
            sid = str(item["sample_id"])
            if "cluster_id" in item and item["cluster_id"] is not None:
                explicit[sid] = self._normalize_cluster_id(item["cluster_id"])
            elif "difficulty_band" in item and item["difficulty_band"] is not None:
                explicit[sid] = self._normalize_cluster_id(item["difficulty_band"])

        if len(explicit) == len(samples):
            return explicit

        # Priority 2: calibration_map => automatic cutting by rank
        if calibration_map is None:
            raise ValueError(
                "Need either explicit sample cluster assignment (cluster_id/difficulty_band) "
                "or calibration_map for automatic initial clustering."
            )

        level_keys: List[Tuple[str, Any]] = []
        for item in samples:
            key = (str(item.get("dataset_id", "unknown")), item.get("raw_difficulty"))
            if key not in level_keys:
                level_keys.append(key)

        known_scores = [float(v) for v in calibration_map.values()]
        default_score = sum(known_scores) / len(known_scores) if known_scores else 0.5

        level_with_score: List[Tuple[Tuple[str, Any], float]] = []
        for key in level_keys:
            score = float(calibration_map.get(key, default_score))
            level_with_score.append((key, score))

        # Higher accuracy => easier cluster.
        level_with_score.sort(key=lambda x: x[1], reverse=True)

        sorted_keys = [k for k, _ in level_with_score]
        chunks = self._split_evenly(sorted_keys, self.cfg.num_clusters)

        level_to_cluster: Dict[Tuple[str, Any], int] = {}
        for cid, chunk in enumerate(chunks):
            for key in chunk:
                level_to_cluster[key] = cid

        assignment: Dict[str, int] = {}
        for item in samples:
            sid = str(item["sample_id"])
            if sid in explicit:
                assignment[sid] = explicit[sid]
                continue
            key = (str(item.get("dataset_id", "unknown")), item.get("raw_difficulty"))
            assignment[sid] = level_to_cluster[key]

        return assignment

    @staticmethod
    def _split_evenly(items: Sequence[Any], k: int) -> List[List[Any]]:
        n = len(items)
        if n == 0:
            return [[] for _ in range(k)]
        chunks: List[List[Any]] = []
        for i in range(k):
            start = (i * n) // k
            end = ((i + 1) * n) // k
            chunks.append(list(items[start:end]))
        return chunks

    def _normalize_cluster_id(self, value: Any) -> int:
        if isinstance(value, int):
            cid = value
        else:
            s = str(value).strip().lower()
            if s in {"easy", "e"}:
                cid = 0
            elif s in {"medium", "mid", "m"}:
                cid = 1 if self.cfg.num_clusters >= 3 else 0
            elif s in {"hard", "h"}:
                cid = self.cfg.num_clusters - 1
            elif s.startswith("cluster_"):
                cid = int(s.split("_", 1)[1])
            else:
                cid = int(s)

        if not (0 <= cid < self.cfg.num_clusters):
            raise ValueError(f"cluster id out of range: {cid}")
        return cid

    def _is_active(self, state: SampleState) -> bool:
        if state.last_update_step < 0:
            return False
        return (self.global_step - state.last_update_step) <= self.cfg.active_window

    def _refresh_cluster_stats(self) -> None:
        for cid, cluster in self.clusters.items():
            active_states = [
                self.samples[sid].s
                for sid in cluster.member_ids
                if self._is_active(self.samples[sid])
            ]
            cluster.active_count = len(active_states)

            if active_states:
                mean = sum(active_states) / len(active_states)
                var = sum((x - mean) ** 2 for x in active_states) / len(active_states)
                std = math.sqrt(var)
                cluster.active_mean = mean
                cluster.active_std = std
                cluster.value = mean
            else:
                cluster.active_mean = 0.0
                cluster.active_std = 0.0
                cluster.value = 0.0

    def _refresh_ucb_and_probs(self) -> None:
        # UCB
        for cid, cluster in self.clusters.items():
            bonus = self.cfg.ucb_beta * math.sqrt(
                2.0 * math.log(self.total_draws + 1.0) / (cluster.sample_count + 1.0)
            )
            cluster.ucb_score = cluster.value + bonus

        # Softmax over UCB
        scores = [self.clusters[cid].ucb_score for cid in range(self.cfg.num_clusters)]
        max_score = max(scores)
        logits = [math.exp((s - max_score) / self.cfg.softmax_tau) for s in scores]
        z = sum(logits)
        probs = [x / z for x in logits]

        # Probability floor
        k = self.cfg.num_clusters
        eps = self.cfg.prob_floor_eps
        floor = eps / k
        probs = [(1.0 - eps) * p + floor for p in probs]

        # Normalize again to avoid floating drift
        z2 = sum(probs)
        probs = [p / z2 for p in probs]

        for cid, p in enumerate(probs):
            self.clusters[cid].prob = p

    def _draw_cluster(self) -> int:
        cids = list(range(self.cfg.num_clusters))
        probs = [self.clusters[cid].prob for cid in cids]
        chosen = self.rng.choices(cids, weights=probs, k=1)[0]

        if self.clusters[chosen].member_ids:
            return chosen

        non_empty = [cid for cid in cids if self.clusters[cid].member_ids]
        if not non_empty:
            raise RuntimeError("No samples available in any cluster.")
        return self.rng.choice(non_empty)

    def sample_batch(self, batch_size: int) -> Tuple[List[str], List[int]]:
        if batch_size <= 0:
            raise ValueError("batch_size must be > 0")

        sample_ids: List[str] = []
        sampled_clusters: List[int] = []

        for _ in range(batch_size):
            cid = self._draw_cluster()
            sid = self.rng.choice(tuple(self.clusters[cid].member_ids))
            sample_ids.append(sid)
            sampled_clusters.append(cid)

            self.clusters[cid].sample_count += 1
            self.total_draws += 1

        # Keep policy current after draw-count updates.
        self._refresh_ucb_and_probs()
        return sample_ids, sampled_clusters

    def update_after_batch(
        self,
        sample_ids: Sequence[str],
        abs_advantages: Sequence[float],
        global_step: int,
    ) -> None:
        if len(sample_ids) != len(abs_advantages):
            raise ValueError("sample_ids and abs_advantages must have the same length")

        self.global_step = int(global_step)

        for sid, adv in zip(sample_ids, abs_advantages):
            key = str(sid)
            if key not in self.samples:
                raise KeyError(f"unknown sample_id in update_after_batch: {key}")

            state = self.samples[key]
            abs_adv = abs(float(adv))
            state.s = self.cfg.decay * state.s + (1.0 - self.cfg.decay) * abs_adv
            state.obs_count += 1
            state.last_update_step = self.global_step

        self._refresh_cluster_stats()
        self._refresh_ucb_and_probs()

    def maybe_rebucket(self, global_step: int) -> List[Dict[str, Any]]:
        self.global_step = int(global_step)

        if self.global_step < self.cfg.warmup_steps:
            return []
        if self.cfg.rebucket_interval <= 0:
            return []
        if self.global_step % self.cfg.rebucket_interval != 0:
            return []
        if self.last_rebucket_step == self.global_step:
            return []

        migrations: List[Dict[str, Any]] = []

        for cid in range(self.cfg.num_clusters):
            cluster = self.clusters[cid]

            eligible_ids = [
                sid
                for sid in cluster.member_ids
                if self._is_active(self.samples[sid])
                and self.samples[sid].obs_count >= self.cfg.min_obs_for_rebucket
            ]
            if not eligible_ids:
                continue

            values = [self.samples[sid].s for sid in eligible_ids]
            mu = sum(values) / len(values)
            var = sum((x - mu) ** 2 for x in values) / len(values)
            sigma = max(math.sqrt(var), 1e-6)

            for sid in eligible_ids:
                state = self.samples[sid]
                delta = state.s - mu

                # Upward deviation => easier migration side.
                if delta > self.cfg.migration_gamma * sigma:
                    state.upward_streak += 1
                else:
                    state.upward_streak = 0

                # Downward deviation => harder migration side (optional).
                if self.cfg.allow_reverse_migration and delta < -self.cfg.migration_gamma * sigma:
                    state.downward_streak += 1
                else:
                    state.downward_streak = 0

                target: Optional[int] = None
                direction: Optional[str] = None

                if state.upward_streak >= self.cfg.migration_consecutive:
                    target = self._easier_cluster(cid)
                    direction = "easier"
                elif self.cfg.allow_reverse_migration and state.downward_streak >= self.cfg.migration_consecutive:
                    target = self._harder_cluster(cid)
                    direction = "harder"

                if target is None or target == cid:
                    continue

                self._migrate_sample(state, from_cluster=cid, to_cluster=target, step=self.global_step)
                migrations.append(
                    {
                        "sample_id": sid,
                        "from_cluster": cid,
                        "to_cluster": target,
                        "direction": direction,
                        "delta": delta,
                        "mu": mu,
                        "sigma": sigma,
                    }
                )

        if migrations:
            self._refresh_cluster_stats()
            self._refresh_ucb_and_probs()

        self.last_rebucket_step = self.global_step
        return migrations

    def _easier_cluster(self, cid: int) -> int:
        if self.cfg.allow_only_neighbor_migration:
            return max(0, cid - 1)
        return 0

    def _harder_cluster(self, cid: int) -> int:
        if self.cfg.allow_only_neighbor_migration:
            return min(self.cfg.num_clusters - 1, cid + 1)
        return self.cfg.num_clusters - 1

    def _migrate_sample(self, state: SampleState, from_cluster: int, to_cluster: int, step: int) -> None:
        if state.sample_id not in self.clusters[from_cluster].member_ids:
            return

        self.clusters[from_cluster].member_ids.remove(state.sample_id)
        self.clusters[to_cluster].member_ids.add(state.sample_id)

        self.clusters[from_cluster].migration_out += 1
        self.clusters[to_cluster].migration_in += 1

        state.migration_history.append(
            {
                "step": int(step),
                "from_cluster": int(from_cluster),
                "to_cluster": int(to_cluster),
            }
        )
        state.cluster_id = int(to_cluster)
        state.upward_streak = 0
        state.downward_streak = 0

    def get_cluster_stats(self) -> Dict[int, Dict[str, Any]]:
        stats: Dict[int, Dict[str, Any]] = {}
        for cid, cluster in self.clusters.items():
            stats[cid] = {
                "cluster_label": self.cluster_labels[cid],
                "size": len(cluster.member_ids),
                "active_size": cluster.active_count,
                "value": cluster.value,
                "active_mean": cluster.active_mean,
                "active_std": cluster.active_std,
                "sample_count": cluster.sample_count,
                "ucb_score": cluster.ucb_score,
                "prob": cluster.prob,
                "migration_in": cluster.migration_in,
                "migration_out": cluster.migration_out,
            }
        return stats

    def get_sample_stats(self, sample_ids: Optional[Iterable[str]] = None) -> Dict[str, Dict[str, Any]]:
        selected = [str(sid) for sid in sample_ids] if sample_ids is not None else list(self.samples.keys())

        out: Dict[str, Dict[str, Any]] = {}
        for sid in selected:
            state = self.samples[sid]
            out[sid] = {
                "dataset_id": state.dataset_id,
                "raw_difficulty": state.raw_difficulty,
                "s": state.s,
                "obs_count": state.obs_count,
                "last_update_step": state.last_update_step,
                "cluster_id": state.cluster_id,
                "cluster_label": self.cluster_labels[state.cluster_id],
                "initial_cluster_id": state.initial_cluster_id,
                "upward_streak": state.upward_streak,
                "downward_streak": state.downward_streak,
                "migration_history": list(state.migration_history),
            }
        return out

    @staticmethod
    def build_calibration_map_from_level_accuracy(
        level_accuracy_rows: Sequence[Mapping[str, Any]],
    ) -> Dict[Tuple[str, Any], float]:
        """Helper: rows with {dataset_id, raw_difficulty, accuracy} -> calibration_map."""
        out: Dict[Tuple[str, Any], float] = {}
        for row in level_accuracy_rows:
            key = (str(row["dataset_id"]), row["raw_difficulty"])
            out[key] = float(row["accuracy"])
        return out
