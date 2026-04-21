#!/usr/bin/env python3
"""Prepare formal random-baseline datasets: math_only / puzzle_only(kk+zebra) / mixed."""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
from datasets import load_dataset

REQUIRED_COLUMNS = ["data_source", "prompt", "ability", "reward_model", "extra_info"]


def _extract_level(value) -> int:
    if value is None:
        return 0
    if isinstance(value, (int, float)):
        return int(value)
    m = re.search(r"\d+", str(value))
    return int(m.group(0)) if m else 0


def _math_prompt(problem: str) -> List[Dict[str, str]]:
    return [
        {
            "role": "user",
            "content": (
                "Solve the following competition math problem step by step. "
                "Put your final answer in \\boxed{...}.\n\n"
                f"Problem: {problem}"
            ),
        }
    ]


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


def _build_math_rows(rows: Sequence[dict], split_name: str) -> pd.DataFrame:
    out = []
    for i, row in enumerate(rows):
        answer = row.get("answer")
        problem = row.get("problem")
        if answer is None or problem is None:
            continue

        level_num = _extract_level(row.get("level"))
        source = f"math500_level_{level_num}" if level_num > 0 else "math500_level_0"

        out.append(
            {
                "data_source": source,
                "prompt": _math_prompt(str(problem)),
                "ability": "math",
                "reward_model": {
                    "style": "rule",
                    "ground_truth": str(answer),
                },
                "extra_info": {
                    "split": split_name,
                    "index": i,
                    "difficulty": float(level_num),
                    "subject": row.get("subject", "unknown"),
                    "level": row.get("level", ""),
                    "unique_id": row.get("unique_id", ""),
                },
            }
        )

    return pd.DataFrame(out, columns=REQUIRED_COLUMNS)


def _sample_math500(train_n: int, val_n: int, seed: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    ds = load_dataset("HuggingFaceH4/MATH-500", split="test")
    total = len(ds)
    need = train_n + val_n
    if need > total:
        raise ValueError(f"Need {need} math rows but dataset has only {total}")

    rng = np.random.default_rng(seed)
    perm = rng.permutation(total)
    train_idx = perm[:train_n]
    val_idx = perm[train_n : train_n + val_n]

    train_rows = [ds[int(i)] for i in train_idx]
    val_rows = [ds[int(i)] for i in val_idx]
    return _build_math_rows(train_rows, "train"), _build_math_rows(val_rows, "val")


def _sample_by_bucket(
    df: pd.DataFrame,
    bucket_col: str,
    bucket_values: Sequence[int],
    train_per_bucket: int,
    val_per_bucket: int,
    seed: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    train_parts = []
    val_parts = []
    rng = np.random.default_rng(seed)

    for b in bucket_values:
        pool = df[df[bucket_col] == b].reset_index(drop=True)
        need = train_per_bucket + val_per_bucket
        if need > len(pool):
            raise ValueError(
                f"Bucket {bucket_col}={b}: need {need}, available {len(pool)}"
            )

        idx = rng.permutation(len(pool))
        train_idx = idx[:train_per_bucket]
        val_idx = idx[train_per_bucket : train_per_bucket + val_per_bucket]

        train_parts.append(pool.iloc[train_idx])
        val_parts.append(pool.iloc[val_idx])

    train_df = pd.concat(train_parts, ignore_index=True)
    val_df = pd.concat(val_parts, ignore_index=True)
    return train_df, val_df


def _save_pair(train_df: pd.DataFrame, val_df: pd.DataFrame, out_dir: Path, seed: int) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    train_df = train_df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    val_df = val_df.sample(frac=1.0, random_state=seed + 1).reset_index(drop=True)
    train_path = out_dir / "train.parquet"
    val_path = out_dir / "val.parquet"
    train_df.to_parquet(train_path, index=False)
    val_df.to_parquet(val_path, index=False)

    print(f"[OK] {out_dir.name}: train={train_df.shape}, val={val_df.shape}")
    print(f"  train sources: {train_df['data_source'].value_counts().to_dict()}")
    print(f"  val sources:   {val_df['data_source'].value_counts().to_dict()}")


def _int_list(raw: str) -> List[int]:
    return [int(x.strip()) for x in raw.split(",") if x.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--kk-train", required=True)
    parser.add_argument("--kk-test", required=True)
    parser.add_argument("--zebra-train", required=True)
    parser.add_argument("--zebra-test", required=True)
    parser.add_argument("--out-root", required=True)

    parser.add_argument("--kk-levels", default="3,4,5,6,7,8")
    parser.add_argument("--zebra-houses", default="3,4,5,6")

    parser.add_argument("--math-train-samples", type=int, default=400)
    parser.add_argument("--math-val-samples", type=int, default=100)

    parser.add_argument("--puzzle-train-per-bucket", type=int, default=160)
    parser.add_argument("--puzzle-val-per-bucket", type=int, default=40)

    parser.add_argument("--mixed-puzzle-train-per-bucket", type=int, default=40)
    parser.add_argument("--mixed-puzzle-val-per-bucket", type=int, default=10)

    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    kk_levels = _int_list(args.kk_levels)
    zebra_houses = _int_list(args.zebra_houses)

    kk_train = pd.read_parquet(args.kk_train)
    kk_test = pd.read_parquet(args.kk_test)
    kk_all = pd.concat([kk_train, kk_test], ignore_index=True)
    kk_all = _to_schema(kk_all)

    kk_keep = [f"kk_logic_{lv}" for lv in kk_levels]
    kk_all = kk_all[kk_all["data_source"].isin(kk_keep)].reset_index(drop=True)
    kk_all["bucket"] = kk_all["data_source"].astype(str).str.extract(r"(\d+)").astype(int)

    zebra_train = pd.read_parquet(args.zebra_train)
    zebra_test = pd.read_parquet(args.zebra_test)
    zebra_all = pd.concat([zebra_train, zebra_test], ignore_index=True)
    zebra_all = _to_schema(zebra_all)
    zebra_all["bucket"] = zebra_all["prompt"].map(_parse_houses)
    zebra_all = zebra_all[zebra_all["bucket"].isin(zebra_houses)].reset_index(drop=True)
    zebra_all["data_source"] = zebra_all["bucket"].map(lambda x: f"zebra_houses_{int(x)}")

    math_train, math_val = _sample_math500(
        train_n=args.math_train_samples,
        val_n=args.math_val_samples,
        seed=args.seed,
    )

    kk_train_df, kk_val_df = _sample_by_bucket(
        kk_all,
        bucket_col="bucket",
        bucket_values=kk_levels,
        train_per_bucket=args.puzzle_train_per_bucket,
        val_per_bucket=args.puzzle_val_per_bucket,
        seed=args.seed + 11,
    )
    zebra_train_df, zebra_val_df = _sample_by_bucket(
        zebra_all,
        bucket_col="bucket",
        bucket_values=zebra_houses,
        train_per_bucket=args.puzzle_train_per_bucket,
        val_per_bucket=args.puzzle_val_per_bucket,
        seed=args.seed + 17,
    )

    puzzle_train = pd.concat([kk_train_df, zebra_train_df], ignore_index=True)
    puzzle_val = pd.concat([kk_val_df, zebra_val_df], ignore_index=True)

    kk_mixed_train, kk_mixed_val = _sample_by_bucket(
        kk_all,
        bucket_col="bucket",
        bucket_values=kk_levels,
        train_per_bucket=args.mixed_puzzle_train_per_bucket,
        val_per_bucket=args.mixed_puzzle_val_per_bucket,
        seed=args.seed + 23,
    )
    zebra_mixed_train, zebra_mixed_val = _sample_by_bucket(
        zebra_all,
        bucket_col="bucket",
        bucket_values=zebra_houses,
        train_per_bucket=args.mixed_puzzle_train_per_bucket,
        val_per_bucket=args.mixed_puzzle_val_per_bucket,
        seed=args.seed + 29,
    )

    mixed_train = pd.concat([math_train, kk_mixed_train, zebra_mixed_train], ignore_index=True)
    mixed_val = pd.concat([math_val, kk_mixed_val, zebra_mixed_val], ignore_index=True)

    for df in [puzzle_train, puzzle_val, mixed_train, mixed_val]:
        if "bucket" in df.columns:
            df.drop(columns=["bucket"], inplace=True)

    out_root = Path(args.out_root)
    _save_pair(math_train, math_val, out_root / "math_only", args.seed)
    _save_pair(puzzle_train, puzzle_val, out_root / "puzzle_only", args.seed)
    _save_pair(mixed_train, mixed_val, out_root / "mixed", args.seed)


if __name__ == "__main__":
    main()
