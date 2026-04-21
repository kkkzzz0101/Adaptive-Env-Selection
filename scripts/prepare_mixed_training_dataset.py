#!/usr/bin/env python3
"""Build mixed dataset for random-sampling GRPO:
- Train mix: kk + zebra + countdown + math_train
- Val mix: kk + zebra + countdown + math_train
- Test mix: kk + zebra + countdown
- Math500 test: from math_test (test_math)
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np
import pandas as pd

REQUIRED_COLUMNS = ["data_source", "prompt", "ability", "reward_model", "extra_info"]


def _to_schema(df: pd.DataFrame) -> pd.DataFrame:
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    return df[REQUIRED_COLUMNS].copy()


def _parse_houses(prompt_obj) -> int | None:
    s = str(prompt_obj)
    m = re.search(r"There are\s+(\d+)\s+houses", s)
    if m:
        return int(m.group(1))
    m = re.search(r"There are\s+(\d+)\s+entities", s)
    if m:
        return int(m.group(1))
    return None


def _parse_countdown_diff(extra_info_obj, data_source: str) -> int | None:
    if isinstance(extra_info_obj, dict) and "difficulty" in extra_info_obj:
        try:
            return int(extra_info_obj["difficulty"])
        except Exception:
            pass

    s = str(extra_info_obj)
    m = re.search(r"'difficulty':\s*(\d+)", s)
    if m:
        return int(m.group(1))

    ds = str(data_source)
    if "very_hard" in ds:
        return 4
    if "hard" in ds:
        return 3
    if "medium" in ds:
        return 2
    if "easy" in ds:
        return 1
    return None


def _parse_math_level(extra_info_obj) -> int | None:
    if isinstance(extra_info_obj, dict):
        if "level" in extra_info_obj:
            m = re.search(r"\d+", str(extra_info_obj["level"]))
            if m:
                return int(m.group(0))
        if "difficulty" in extra_info_obj:
            try:
                return int(extra_info_obj["difficulty"])
            except Exception:
                pass

    s = str(extra_info_obj)
    m = re.search(r"'level':\s*'[^']*(\d+)" , s)
    if m:
        return int(m.group(1))
    m = re.search(r"'difficulty':\s*(\d+)" , s)
    if m:
        return int(m.group(1))
    return None


def _sample_three_way(
    pool: pd.DataFrame,
    train_n: int,
    val_n: int,
    test_n: int,
    seed: int,
    label: str,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    n = len(pool)
    if n == 0:
        raise ValueError(f"Empty pool for {label}")

    # Keep validation/test unique if possible; train may use replacement when scarce.
    vt_need = val_n + test_n
    if vt_need <= n:
        perm = rng.permutation(n)
        val_idx = perm[:val_n]
        test_idx = perm[val_n:val_n + test_n]
        remaining_idx = perm[val_n + test_n:]
        remaining = pool.iloc[remaining_idx].reset_index(drop=True)
    else:
        # Extremely scarce edge case: sample val/test with replacement.
        val_idx = rng.integers(0, n, size=val_n)
        test_idx = rng.integers(0, n, size=test_n)
        remaining = pool.copy().reset_index(drop=True)

    val_df = pool.iloc[val_idx].reset_index(drop=True)
    test_df = pool.iloc[test_idx].reset_index(drop=True)

    if len(remaining) >= train_n:
        tr_perm = rng.permutation(len(remaining))[:train_n]
        train_df = remaining.iloc[tr_perm].reset_index(drop=True)
        replacement = False
    else:
        source = remaining if len(remaining) > 0 else pool.reset_index(drop=True)
        tr_idx = rng.integers(0, len(source), size=train_n)
        train_df = source.iloc[tr_idx].reset_index(drop=True)
        replacement = True

    if replacement:
        print(f"[WARN] {label}: insufficient unique rows; train sampled with replacement.")

    return train_df, val_df, test_df


def _sample_two_way(
    pool: pd.DataFrame,
    train_n: int,
    val_n: int,
    seed: int,
    label: str,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    need = train_n + val_n
    n = len(pool)
    if n == 0:
        raise ValueError(f"Empty pool for {label}")

    if need <= n:
        perm = rng.permutation(n)
        train_idx = perm[:train_n]
        val_idx = perm[train_n:train_n + val_n]
        return pool.iloc[train_idx].reset_index(drop=True), pool.iloc[val_idx].reset_index(drop=True)

    # Fall back: val unique as much as possible, train with replacement.
    val_take = min(val_n, n)
    perm = rng.permutation(n)
    val_df = pool.iloc[perm[:val_take]].reset_index(drop=True)
    if val_take < val_n:
        extra_idx = rng.integers(0, n, size=(val_n - val_take))
        val_df = pd.concat([val_df, pool.iloc[extra_idx].reset_index(drop=True)], ignore_index=True)

    train_idx = rng.integers(0, n, size=train_n)
    train_df = pool.iloc[train_idx].reset_index(drop=True)
    print(f"[WARN] {label}: insufficient unique rows; train sampled with replacement.")
    return train_df, val_df


def _sample_family_three_way(
    df: pd.DataFrame,
    bucket_col: str,
    bucket_values: Sequence[int],
    train_per_bucket: int,
    val_per_bucket: int,
    test_per_bucket: int,
    seed: int,
    family_name: str,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    tr_parts, va_parts, te_parts = [], [], []
    for i, b in enumerate(bucket_values):
        pool = df[df[bucket_col] == b].reset_index(drop=True)
        tr, va, te = _sample_three_way(
            pool,
            train_n=train_per_bucket,
            val_n=val_per_bucket,
            test_n=test_per_bucket,
            seed=seed + i,
            label=f"{family_name}:{b}",
        )
        tr_parts.append(tr)
        va_parts.append(va)
        te_parts.append(te)

    return (
        pd.concat(tr_parts, ignore_index=True),
        pd.concat(va_parts, ignore_index=True),
        pd.concat(te_parts, ignore_index=True),
    )


def _sample_math_two_way(
    df: pd.DataFrame,
    level_col: str,
    levels: Sequence[int],
    train_per_level: int,
    val_per_level: int,
    seed: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    tr_parts, va_parts = [], []
    for i, lv in enumerate(levels):
        pool = df[df[level_col] == lv].reset_index(drop=True)
        tr, va = _sample_two_way(
            pool,
            train_n=train_per_level,
            val_n=val_per_level,
            seed=seed + i,
            label=f"math_level:{lv}",
        )
        tr_parts.append(tr)
        va_parts.append(va)

    return pd.concat(tr_parts, ignore_index=True), pd.concat(va_parts, ignore_index=True)


def _normalize_reward_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    def _norm_rm(rm):
        if not isinstance(rm, dict):
            return {'style': 'rule', 'ground_truth': str(rm)}
        gt = rm.get('ground_truth')
        if isinstance(gt, (dict, list)):
            gt = json.dumps(gt, ensure_ascii=False)
        elif gt is None:
            gt = ''
        else:
            gt = str(gt)
        return {'style': rm.get('style', 'rule'), 'ground_truth': gt}

    out['reward_model'] = out['reward_model'].map(_norm_rm)
    return out

def _save_df(df: pd.DataFrame, path: Path, seed: int, title: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df = _normalize_reward_columns(df)
    out = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    out.to_parquet(path, index=False)
    print(f"\n[OK] {title}: {path} shape={out.shape}")
    print(out["data_source"].value_counts().to_string())


def _int_list(raw: str) -> List[int]:
    return [int(x.strip()) for x in raw.split(",") if x.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--kk-train", required=True)
    parser.add_argument("--kk-test", required=True)
    parser.add_argument("--zebra-train", required=True)
    parser.add_argument("--zebra-test", required=True)
    parser.add_argument("--countdown-train", required=True)
    parser.add_argument("--countdown-test", required=True)
    parser.add_argument("--math-train", required=True)
    parser.add_argument("--math-test", required=True)
    parser.add_argument("--out-root", required=True)

    parser.add_argument("--kk-levels", default="3,4,5,6,7,8")
    parser.add_argument("--zebra-houses", default="3,4,5,6")
    parser.add_argument("--countdown-diffs", default="1,2,3,4")
    parser.add_argument("--math-levels", default="1,2,3,4,5")

    parser.add_argument("--train-per-bucket", type=int, default=240)
    parser.add_argument("--val-per-bucket", type=int, default=30)
    parser.add_argument("--test-per-bucket", type=int, default=80)
    parser.add_argument("--math-train-per-level", type=int, default=300)
    parser.add_argument("--math-val-per-level", type=int, default=30)

    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    kk_levels = _int_list(args.kk_levels)
    zebra_houses = _int_list(args.zebra_houses)
    countdown_diffs = _int_list(args.countdown_diffs)
    math_levels = _int_list(args.math_levels)

    kk_all = pd.concat([pd.read_parquet(args.kk_train), pd.read_parquet(args.kk_test)], ignore_index=True)
    kk_all = _to_schema(kk_all)
    kk_all = kk_all[kk_all["data_source"].isin([f"kk_logic_{x}" for x in kk_levels])].reset_index(drop=True)
    kk_all["bucket"] = kk_all["data_source"].astype(str).str.extract(r"(\d+)").astype(int)

    zebra_all = pd.concat([pd.read_parquet(args.zebra_train), pd.read_parquet(args.zebra_test)], ignore_index=True)
    zebra_all = _to_schema(zebra_all)
    zebra_all["bucket"] = zebra_all["prompt"].map(_parse_houses)
    zebra_all = zebra_all[zebra_all["bucket"].isin(zebra_houses)].reset_index(drop=True)
    zebra_all["data_source"] = zebra_all["bucket"].map(lambda x: f"zebra_houses_{int(x)}")

    countdown_all = pd.concat([pd.read_parquet(args.countdown_train), pd.read_parquet(args.countdown_test)], ignore_index=True)
    countdown_all = _to_schema(countdown_all)
    countdown_all["bucket"] = [
        _parse_countdown_diff(ei, ds)
        for ei, ds in zip(countdown_all["extra_info"], countdown_all["data_source"])
    ]
    countdown_all = countdown_all[countdown_all["bucket"].isin(countdown_diffs)].reset_index(drop=True)
    countdown_all["data_source"] = countdown_all["bucket"].map(lambda x: f"countdown_diff_{int(x)}")

    math_train_all = _to_schema(pd.read_parquet(args.math_train))
    math_train_all["level_bucket"] = math_train_all["extra_info"].map(_parse_math_level)
    math_train_all = math_train_all[math_train_all["level_bucket"].isin(math_levels)].reset_index(drop=True)
    math_train_all["data_source"] = math_train_all["level_bucket"].map(lambda x: f"math_train_level_{int(x)}")

    kk_tr, kk_va, kk_te = _sample_family_three_way(
        kk_all,
        bucket_col="bucket",
        bucket_values=kk_levels,
        train_per_bucket=args.train_per_bucket,
        val_per_bucket=args.val_per_bucket,
        test_per_bucket=args.test_per_bucket,
        seed=args.seed + 10,
        family_name="kk",
    )
    zebra_tr, zebra_va, zebra_te = _sample_family_three_way(
        zebra_all,
        bucket_col="bucket",
        bucket_values=zebra_houses,
        train_per_bucket=args.train_per_bucket,
        val_per_bucket=args.val_per_bucket,
        test_per_bucket=args.test_per_bucket,
        seed=args.seed + 20,
        family_name="zebra",
    )
    countdown_tr, countdown_va, countdown_te = _sample_family_three_way(
        countdown_all,
        bucket_col="bucket",
        bucket_values=countdown_diffs,
        train_per_bucket=args.train_per_bucket,
        val_per_bucket=args.val_per_bucket,
        test_per_bucket=args.test_per_bucket,
        seed=args.seed + 30,
        family_name="countdown",
    )
    math_tr, math_va = _sample_math_two_way(
        math_train_all,
        level_col="level_bucket",
        levels=math_levels,
        train_per_level=args.math_train_per_level,
        val_per_level=args.math_val_per_level,
        seed=args.seed + 40,
    )

    train_df = pd.concat([kk_tr, zebra_tr, countdown_tr, math_tr], ignore_index=True)
    val_df = pd.concat([kk_va, zebra_va, countdown_va, math_va], ignore_index=True)
    test_df = pd.concat([kk_te, zebra_te, countdown_te], ignore_index=True)

    for df in [train_df, val_df, test_df]:
        for col in ["bucket", "level_bucket"]:
            if col in df.columns:
                df.drop(columns=[col], inplace=True)

    math_test = _to_schema(pd.read_parquet(args.math_test))
    math500_test = math_test[math_test["data_source"] == "test_math"].copy().reset_index(drop=True)
    math500_test["data_source"] = "math500_test"

    out_root = Path(args.out_root)
    _save_df(train_df, out_root / "mixed" / "train.parquet", args.seed, "mixed train")
    _save_df(val_df, out_root / "mixed" / "val.parquet", args.seed + 1, "mixed val")
    _save_df(test_df, out_root / "mixed" / "test.parquet", args.seed + 2, "mixed test (kk/zebra/countdown)")
    _save_df(math500_test, out_root / "mixed" / "math500_test.parquet", args.seed + 3, "math500 test")


if __name__ == "__main__":
    main()
