#!/usr/bin/env python3
"""Build mixed DUMP smoke dataset: existing KK subset + MATH-500 subset."""

from __future__ import annotations

import argparse
import re
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from datasets import load_dataset


REQUIRED_COLUMNS = ["data_source", "prompt", "ability", "reward_model", "extra_info"]


def _extract_level(value) -> int:
    if value is None:
        return 0
    if isinstance(value, (int, float)):
        return int(value)
    match = re.search(r"\d+", str(value))
    return int(match.group(0)) if match else 0


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


def _to_required_schema(df: pd.DataFrame) -> pd.DataFrame:
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    return df[REQUIRED_COLUMNS].copy()


def _build_math500_rows(split_name: str, rows: List[dict], start_index: int = 0) -> pd.DataFrame:
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
                    "index": start_index + i,
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
        raise ValueError(f"Requested {need} MATH-500 rows but only {total} available")

    rng = np.random.default_rng(seed)
    perm = rng.permutation(total)
    train_idx = perm[:train_n]
    val_idx = perm[train_n : train_n + val_n]

    train_rows = [ds[int(i)] for i in train_idx]
    val_rows = [ds[int(i)] for i in val_idx]

    return _build_math500_rows("train", train_rows), _build_math500_rows("val", val_rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build mixed smoke dataset with MATH-500")
    parser.add_argument("--kk-train", required=True, help="Path to existing KK train parquet")
    parser.add_argument("--kk-val", required=True, help="Path to existing KK val parquet")
    parser.add_argument("--out-train", required=True, help="Output train parquet path")
    parser.add_argument("--out-val", required=True, help="Output val parquet path")
    parser.add_argument("--math-train-samples", type=int, default=24)
    parser.add_argument("--math-val-samples", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    kk_train = _to_required_schema(pd.read_parquet(args.kk_train))
    kk_val = _to_required_schema(pd.read_parquet(args.kk_val))

    # Idempotent rebuild: keep base non-math rows and regenerate math500 slice each run.
    kk_train = kk_train[~kk_train['data_source'].astype(str).str.startswith('math500')].reset_index(drop=True)
    kk_val = kk_val[~kk_val['data_source'].astype(str).str.startswith('math500')].reset_index(drop=True)

    math_train, math_val = _sample_math500(
        train_n=args.math_train_samples,
        val_n=args.math_val_samples,
        seed=args.seed,
    )

    mixed_train = pd.concat([kk_train, math_train], ignore_index=True)
    mixed_val = pd.concat([kk_val, math_val], ignore_index=True)

    mixed_train = mixed_train.sample(frac=1.0, random_state=args.seed).reset_index(drop=True)
    mixed_val = mixed_val.sample(frac=1.0, random_state=args.seed + 1).reset_index(drop=True)

    mixed_train.to_parquet(args.out_train, index=False)
    mixed_val.to_parquet(args.out_val, index=False)

    print("[OK] mixed train:", args.out_train, "shape=", mixed_train.shape)
    print("[OK] mixed val:", args.out_val, "shape=", mixed_val.shape)
    print("[train data_source]", mixed_train["data_source"].value_counts().to_dict())
    print("[val data_source]", mixed_val["data_source"].value_counts().to_dict())


if __name__ == "__main__":
    main()
