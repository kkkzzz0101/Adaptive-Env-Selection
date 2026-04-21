#!/usr/bin/env python3
"""Prepare kk-only / math500-only / mixed datasets for random baseline runs."""

from __future__ import annotations

import argparse
import re
from pathlib import Path
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


def _build_math_rows(split_name: str, rows: List[dict]) -> pd.DataFrame:
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
    return _build_math_rows("train", train_rows), _build_math_rows("val", val_rows)


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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--kk-train-source", required=True)
    parser.add_argument("--kk-val-source", required=True)
    parser.add_argument("--out-root", required=True)
    parser.add_argument("--math-train-samples", type=int, default=32)
    parser.add_argument("--math-val-samples", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    kk_train = _to_schema(pd.read_parquet(args.kk_train_source))
    kk_val = _to_schema(pd.read_parquet(args.kk_val_source))

    # Keep only kk rows from potentially mixed sources.
    kk_train = kk_train[~kk_train["data_source"].astype(str).str.startswith("math500")].reset_index(drop=True)
    kk_val = kk_val[~kk_val["data_source"].astype(str).str.startswith("math500")].reset_index(drop=True)

    math_train, math_val = _sample_math500(
        train_n=args.math_train_samples,
        val_n=args.math_val_samples,
        seed=args.seed,
    )

    out_root = Path(args.out_root)

    _save_pair(kk_train, kk_val, out_root / "kk_only", args.seed)
    _save_pair(math_train, math_val, out_root / "math_only", args.seed)
    _save_pair(
        pd.concat([kk_train, math_train], ignore_index=True),
        pd.concat([kk_val, math_val], ignore_index=True),
        out_root / "mixed",
        args.seed,
    )


if __name__ == "__main__":
    main()
