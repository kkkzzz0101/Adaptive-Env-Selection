#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import json
import re
from pathlib import Path
from typing import Any, Tuple

import numpy as np
import pandas as pd

from prompt_protocol import append_protocol_text, extract_countdown_target, infer_dataset_from_data_source

REQUIRED_COLUMNS = ["data_source", "prompt", "ability", "reward_model", "extra_info"]


def _coerce_prompt_messages(prompt: Any) -> list[dict[str, Any]] | None:
    if isinstance(prompt, list):
        return prompt

    if hasattr(prompt, "tolist") and not isinstance(prompt, (str, bytes)):
        try:
            obj = prompt.tolist()
            if isinstance(obj, list):
                return obj
        except Exception:
            pass

    if isinstance(prompt, str):
        text = prompt.strip()
        if text.startswith('[') and text.endswith(']'):
            for parser in (json.loads, ast.literal_eval):
                try:
                    obj = parser(text)
                    if isinstance(obj, list):
                        return obj
                except Exception:
                    continue
    return None


def _enforce_final_boxed_prompt(prompt: Any, data_source: Any, reward_model: Any) -> Any:
    msgs = _coerce_prompt_messages(prompt)
    if not msgs:
        return prompt

    changed = False
    msgs = list(msgs)
    for i in range(len(msgs) - 1, -1, -1):
        msg = msgs[i]
        if not isinstance(msg, dict):
            continue
        if str(msg.get('role', '')).lower() != 'user':
            continue

        content = msg.get('content')
        if not isinstance(content, str):
            break

        dataset = infer_dataset_from_data_source(data_source)
        target = extract_countdown_target(reward_model) if dataset == 'countdown' else None
        updated = append_protocol_text(content, dataset=dataset, target=target)
        if updated != content:
            msg_new = dict(msg)
            msg_new['content'] = updated
            msgs[i] = msg_new
            changed = True
        break

    if not changed:
        return prompt

    if isinstance(prompt, np.ndarray):
        try:
            return np.array(msgs, dtype=object)
        except Exception:
            return msgs
    return msgs


def _to_schema(df: pd.DataFrame) -> pd.DataFrame:
    miss = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if miss:
        raise ValueError(f"Missing required columns: {miss}")
    return df[REQUIRED_COLUMNS].copy()


def _normalize_reward_model(obj: Any) -> Any:
    if not isinstance(obj, dict):
        return obj
    out = dict(obj)
    gt = out.get('ground_truth')
    if isinstance(gt, str):
        s = gt.strip()
        if (s.startswith('{') and s.endswith('}')) or (s.startswith('[') and s.endswith(']')):
            try:
                out['ground_truth'] = json.loads(s)
            except Exception:
                pass
    return out


def _parse_level_from_ds(ds: str) -> int | None:
    s = ds.lower()
    if "very_hard" in s:
        return 4
    if "hard" in s and "very_hard" not in s:
        return 3
    if "medium" in s:
        return 2
    if "easy" in s:
        return 1
    return None


def _difficulty(row: pd.Series) -> int | None:
    ds = str(row.get("data_source", ""))
    lv = _parse_level_from_ds(ds)
    if lv is not None:
        return lv
    ex = row.get("extra_info")
    if isinstance(ex, dict) and "difficulty" in ex:
        try:
            return int(ex["difficulty"])
        except Exception:
            pass
    return None


def _math_level(ex: Any) -> int | None:
    if isinstance(ex, dict):
        if "difficulty" in ex:
            try:
                return int(ex["difficulty"])
            except Exception:
                pass
        if "level" in ex:
            m = re.search(r"\d+", str(ex["level"]))
            if m:
                return int(m.group(0))
    s = str(ex)
    m = re.search(r"'difficulty':\s*(\d+)", s)
    if m:
        return int(m.group(1))
    m = re.search(r"'level':\s*'[^']*(\d+)", s)
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
    n = len(pool)
    if n == 0:
        raise ValueError(f"Empty pool for {label}")
    rng = np.random.default_rng(seed)

    need = train_n + val_n + test_n
    if need <= n:
        perm = rng.permutation(n)
        tr = pool.iloc[perm[:train_n]].reset_index(drop=True)
        va = pool.iloc[perm[train_n:train_n + val_n]].reset_index(drop=True)
        te = pool.iloc[perm[train_n + val_n:train_n + val_n + test_n]].reset_index(drop=True)
        return tr, va, te

    tr_idx = rng.integers(0, n, size=train_n)
    va_idx = rng.integers(0, n, size=val_n)
    te_idx = rng.integers(0, n, size=test_n)
    print(f"[WARN] {label}: insufficient unique rows ({n} < {need}), sampling with replacement")
    return (
        pool.iloc[tr_idx].reset_index(drop=True),
        pool.iloc[va_idx].reset_index(drop=True),
        pool.iloc[te_idx].reset_index(drop=True),
    )


def _sample_two_way(
    pool: pd.DataFrame,
    train_n: int,
    val_n: int,
    seed: int,
    label: str,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    n = len(pool)
    if n == 0:
        raise ValueError(f"Empty pool for {label}")
    rng = np.random.default_rng(seed)
    need = train_n + val_n
    if need <= n:
        perm = rng.permutation(n)
        tr = pool.iloc[perm[:train_n]].reset_index(drop=True)
        va = pool.iloc[perm[train_n:train_n + val_n]].reset_index(drop=True)
        return tr, va

    tr_idx = rng.integers(0, n, size=train_n)
    va_idx = rng.integers(0, n, size=val_n)
    print(f"[WARN] {label}: insufficient unique rows ({n} < {need}), sampling with replacement")
    return pool.iloc[tr_idx].reset_index(drop=True), pool.iloc[va_idx].reset_index(drop=True)


def _normalize(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    def norm_rm(rm: Any) -> dict:
        if isinstance(rm, dict):
            gt = rm.get("ground_truth", "")
            if isinstance(gt, (dict, list)):
                gt = json.dumps(gt, ensure_ascii=False)
            elif gt is None:
                gt = ""
            else:
                gt = str(gt)
            return {"style": str(rm.get("style", "rule")), "ground_truth": gt}
        return {"style": "rule", "ground_truth": "" if rm is None else str(rm)}

    def norm_ex(ex: Any) -> dict:
        if isinstance(ex, dict):
            fixed = {}
            for k, v in ex.items():
                if isinstance(v, (int, float, bool, str)) or v is None:
                    fixed[k] = '' if v is None else v
                else:
                    fixed[k] = json.dumps(v, ensure_ascii=False) if isinstance(v, (dict, list)) else str(v)
            return fixed
        if isinstance(ex, (int, float, bool, str)):
            return {'raw': ex}
        return {'raw': '' if ex is None else str(ex)}

    out['prompt'] = out.apply(
        lambda r: _enforce_final_boxed_prompt(r['prompt'], r.get('data_source', ''), r.get('reward_model')),
        axis=1,
    )
    out['reward_model'] = out['reward_model'].map(norm_rm)
    out['extra_info'] = out['extra_info'].map(norm_ex)
    return out


def _save(df: pd.DataFrame, path: Path, seed: int, title: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    out = _normalize(df).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    out.to_parquet(path, index=False)
    print(f"[OK] {title}: {path} shape={out.shape}")
    print(out['data_source'].value_counts().sort_index().to_string())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--sec-root', default='/root/adaptive env selection/references/sec/data')
    ap.add_argument('--out-root', default='/root/adaptive env selection/experiments/baselines/data_sec4')
    ap.add_argument('--countdown-levels', default='1,2,3,4')
    ap.add_argument('--zebra-levels', default='1,2,3,4')
    ap.add_argument('--arc-levels', default='1,2,3,4')
    ap.add_argument('--math-levels', default='1,2,3,4,5')

    ap.add_argument('--train-per-bucket', type=int, default=120)
    ap.add_argument('--val-per-bucket', type=int, default=20)
    ap.add_argument('--test-per-bucket', type=int, default=60)

    ap.add_argument('--math-train-per-level', type=int, default=300)
    ap.add_argument('--math-val-per-level', type=int, default=30)

    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()

    sec = Path(args.sec_root)
    out_root = Path(args.out_root)

    c_levels = [int(x) for x in args.countdown_levels.split(',') if x.strip()]
    z_levels = [int(x) for x in args.zebra_levels.split(',') if x.strip()]
    a_levels = [int(x) for x in args.arc_levels.split(',') if x.strip()]
    m_levels = [int(x) for x in args.math_levels.split(',') if x.strip()]

    countdown = pd.concat([
        pd.read_parquet(sec / 'countdown' / 'train.parquet'),
        pd.read_parquet(sec / 'countdown' / 'test.parquet'),
    ], ignore_index=True)
    zebra = pd.concat([

        pd.read_parquet(sec / 'zebra' / 'train.parquet'),
        pd.read_parquet(sec / 'zebra' / 'test.parquet'),
    ], ignore_index=True)
    arc = pd.concat([
        pd.read_parquet(sec / 'arc' / 'train.parquet'),
        pd.read_parquet(sec / 'arc' / 'test.parquet'),
    ], ignore_index=True)
    math_train = pd.read_parquet(sec / 'math' / 'math_train.parquet')
    math_test = pd.read_parquet(sec / 'math' / 'math_test.parquet')

    countdown = _to_schema(countdown)
    zebra = _to_schema(zebra)
    arc = _to_schema(arc)
    math_train = _to_schema(math_train)
    math_test = _to_schema(math_test)

    countdown['bucket'] = countdown.apply(_difficulty, axis=1)
    for _df in (countdown, zebra, arc, math_train, math_test):
        _df['reward_model'] = _df['reward_model'].map(_normalize_reward_model)

    zebra['bucket'] = zebra.apply(_difficulty, axis=1)
    arc['bucket'] = arc.apply(_difficulty, axis=1)
    math_train['level_bucket'] = math_train['extra_info'].map(_math_level)

    countdown = countdown[countdown['bucket'].isin(c_levels)].reset_index(drop=True)
    zebra = zebra[zebra['bucket'].isin(z_levels)].reset_index(drop=True)
    arc = arc[arc['bucket'].isin(a_levels)].reset_index(drop=True)
    math_train = math_train[math_train['level_bucket'].isin(m_levels)].reset_index(drop=True)

    countdown['data_source'] = 'countdown_train'
    zebra['data_source'] = 'zebra_train'
    arc['data_source'] = 'arc_train'
    math_train['data_source'] = 'math_train'

    tr_parts, va_parts, te_parts = [], [], []
    for i, lv in enumerate(c_levels):
        tr, va, te = _sample_three_way(
            countdown[countdown['bucket'] == lv].reset_index(drop=True),
            args.train_per_bucket, args.val_per_bucket, args.test_per_bucket,
            args.seed + 10 + i, f'countdown:{lv}'
        )
        tr_parts.append(tr); va_parts.append(va); te_parts.append(te)
    for i, lv in enumerate(z_levels):
        tr, va, te = _sample_three_way(
            zebra[zebra['bucket'] == lv].reset_index(drop=True),
            args.train_per_bucket, args.val_per_bucket, args.test_per_bucket,
            args.seed + 20 + i, f'zebra:{lv}'
        )
        tr_parts.append(tr); va_parts.append(va); te_parts.append(te)
    for i, lv in enumerate(a_levels):
        tr, va, te = _sample_three_way(
            arc[arc['bucket'] == lv].reset_index(drop=True),
            args.train_per_bucket, args.val_per_bucket, args.test_per_bucket,
            args.seed + 30 + i, f'arc:{lv}'
        )
        tr_parts.append(tr); va_parts.append(va); te_parts.append(te)

    math_tr_parts, math_va_parts = [], []
    for i, lv in enumerate(m_levels):
        tr, va = _sample_two_way(
            math_train[math_train['level_bucket'] == lv].reset_index(drop=True),
            args.math_train_per_level, args.math_val_per_level,
            args.seed + 40 + i, f'math:{lv}'
        )
        math_tr_parts.append(tr); math_va_parts.append(va)

    train_df = pd.concat(tr_parts + math_tr_parts, ignore_index=True)
    val_df = pd.concat(va_parts + math_va_parts, ignore_index=True)
    test_df = pd.concat(te_parts, ignore_index=True)

    # Remap data_source to SEC main_ppo expected routing names.
    train_df.loc[train_df['data_source'].str.startswith('countdown'), 'data_source'] = 'countdown_train'
    train_df.loc[train_df['data_source'].str.startswith('zebra'), 'data_source'] = 'zebra_train'
    train_df.loc[train_df['data_source'].str.startswith('arc'), 'data_source'] = 'arc_train'
    train_df.loc[train_df['data_source'].str.startswith('math_train'), 'data_source'] = 'math_train'

    val_df.loc[val_df['data_source'].str.startswith('countdown'), 'data_source'] = 'countdown_train'
    val_df.loc[val_df['data_source'].str.startswith('zebra'), 'data_source'] = 'zebra_train'
    val_df.loc[val_df['data_source'].str.startswith('arc'), 'data_source'] = 'arc_train'
    val_df.loc[val_df['data_source'].str.startswith('math_train'), 'data_source'] = 'math_train'

    test_df.loc[test_df['data_source'].str.startswith('countdown'), 'data_source'] = 'countdown_test'
    test_df.loc[test_df['data_source'].str.startswith('zebra'), 'data_source'] = 'zebra_test'
    test_df.loc[test_df['data_source'].str.startswith('arc'), 'data_source'] = 'arc_test'

    for df in (train_df, val_df, test_df):
        for c in ['bucket', 'level_bucket']:
            if c in df.columns:
                df.drop(columns=[c], inplace=True)

    math500 = math_test[math_test['data_source'] == 'test_math'].copy().reset_index(drop=True)
    math500['data_source'] = 'math500_test'

    _save(train_df, out_root / 'mixed' / 'train.parquet', args.seed, 'sec4 mixed train')
    _save(val_df, out_root / 'mixed' / 'val.parquet', args.seed + 1, 'sec4 mixed val')
    _save(test_df, out_root / 'mixed' / 'test.parquet', args.seed + 2, 'sec4 mixed test')
    _save(math500, out_root / 'mixed' / 'math500_test.parquet', args.seed + 3, 'math500 test')


if __name__ == '__main__':
    main()
