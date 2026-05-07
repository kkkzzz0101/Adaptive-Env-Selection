#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import json
import os
import random
import re
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(os.environ.get('AES_ROOT', Path(__file__).resolve().parents[1])).resolve()
SEC_VERL = ROOT / 'references' / 'sec' / 'verl'
if str(SEC_VERL) not in sys.path:
    sys.path.insert(0, str(SEC_VERL))

from verl.utils.reward_score.utils import extract_solution, last_boxed_only_string, remove_boxed
from verl.utils.reward_score.deepscaler import get_deepscaler_reward_fn
from verl.utils.reward_score.deepscaler.math_utils import extract_answer as extract_math_answer
from verl.utils.reward_score.zebra import get_zebra_compute_score
from verl.utils.reward_score.arc import get_arc_compute_score
from verl.utils.reward_score.countdown import get_countdown_compute_score

from prompt_protocol import append_protocol_text, extract_countdown_target
from countdown_validator import validate_countdown_expression


@dataclass
class ProbeRow:
    dataset: str
    difficulty: int
    idx: int
    rollout_id: int
    format_valid: int
    parse_success: int
    pass_score: float
    reward_score: float
    parsed_answer: str | None
    raw_parsed_answer: str | None
    canonicalized: int
    ground_truth: Any
    response: str
    countdown_numbers_ok: int | None = None
    countdown_target_ok: int | None = None


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description='SEC inference probe by dataset x difficulty')
    p.add_argument('--dataset', choices=['countdown', 'zebra', 'arc', 'math'], required=True)
    p.add_argument('--model-path', default=os.environ.get('AES_MODEL_PATH', 'Qwen/Qwen2.5-1.5B-Instruct'))
    p.add_argument('--sec-root', default=str(ROOT / 'references' / 'sec' / 'data'))
    p.add_argument('--split', default='test')
    p.add_argument('--n-per-difficulty', type=int, default=20)
    p.add_argument('--rollouts', type=int, default=1)
    p.add_argument('--max-new-tokens', type=int, default=220)
    p.add_argument('--temperature', type=float, default=0.0)
    p.add_argument('--top-p', type=float, default=1.0)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--out-dir', default=str(ROOT / 'experiments' / 'baselines' / 'results_512' / 'sec_probe'))
    return p.parse_args()


def _data_path(sec_root: Path, dataset: str, split: str) -> Path:
    if dataset == 'math':
        return sec_root / 'math' / ('math_train.parquet' if split == 'train' else 'math_test.parquet')
    return sec_root / dataset / f'{split}.parquet'


def _difficulty(extra_info: Any, data_source: str) -> int:
    if isinstance(extra_info, dict) and 'difficulty' in extra_info:
        try:
            return int(extra_info['difficulty'])
        except Exception:
            pass
    text = str(data_source).lower()
    if 'easy' in text:
        return 1
    if 'medium' in text:
        return 2
    if 'hard' in text and 'very_hard' not in text:
        return 3
    if 'very_hard' in text:
        return 4
    return 1


def _messages(prompt: Any) -> list[dict[str, str]]:
    if isinstance(prompt, list):
        return prompt

    # Parquet may store object arrays (numpy.ndarray) for nested prompts.
    if hasattr(prompt, 'tolist') and not isinstance(prompt, (str, bytes)):
        try:
            obj = prompt.tolist()
            if isinstance(obj, list):
                return obj
        except Exception:
            pass

    if isinstance(prompt, str):
        s = prompt.strip()
        if s.startswith('[') and s.endswith(']'):
            for parser in (json.loads, ast.literal_eval):
                try:
                    obj = parser(s)
                    if isinstance(obj, list):
                        return obj
                except Exception:
                    continue
        return [{'role': 'user', 'content': prompt}]
    return [{'role': 'user', 'content': str(prompt)}]


def _apply_prompt_protocol(msgs: list[dict[str, str]], dataset: str, target: str | None = None) -> list[dict[str, str]]:
    out = list(msgs)
    for i in range(len(out) - 1, -1, -1):
        msg = out[i]
        if not isinstance(msg, dict):
            continue
        if str(msg.get('role', '')).lower() != 'user':
            continue

        content = msg.get('content')
        if not isinstance(content, str):
            break

        updated = append_protocol_text(content, dataset=dataset, target=target)
        if updated != content:
            msg_new = dict(msg)
            msg_new['content'] = updated
            out[i] = msg_new
        break
    return out


def _scorers(dataset: str):
    if dataset == 'zebra':
        return get_zebra_compute_score(correct_score=1.0, format_score=0.0), get_zebra_compute_score(correct_score=1.0, format_score=0.1)
    if dataset == 'arc':
        return get_arc_compute_score(correct_score=1.0, format_score=0.0), get_arc_compute_score(correct_score=1.0, format_score=0.1)
    if dataset == 'countdown':
        return get_countdown_compute_score(correct_score=1.0, format_score=0.0), get_countdown_compute_score(correct_score=1.0, format_score=0.1)
    return get_deepscaler_reward_fn(correct_reward=1.0, format_reward=0.0), get_deepscaler_reward_fn(correct_reward=1.0, format_reward=0.1)


def _assistant_segment(full_text: str) -> str:
    if '<|im_start|>assistant' in full_text:
        return full_text.split('<|im_start|>assistant', 1)[1]
    if 'Assistant:' in full_text:
        return full_text.split('Assistant:', 1)[1]
    return full_text


def _parse_answer(dataset: str, full_text: str) -> str | None:
    if dataset in {'zebra', 'arc', 'countdown'}:
        return extract_solution(full_text)

    seg = _assistant_segment(full_text)
    return extract_math_answer(seg[-600:])


def _extract_boxed_anywhere(text: str) -> str | None:
    boxed = last_boxed_only_string(text)
    if boxed is None:
        return None
    inner = remove_boxed(boxed)
    if inner is None:
        return None
    ans = str(inner).strip()
    return ans if ans else None


def _extract_name_pool(prompt_text: str) -> list[str]:
    m = re.search(r'unique name[s]?:\s*([^\n]+)', prompt_text, flags=re.IGNORECASE)
    if not m:
        return []
    pool = []
    for part in re.split(r'[,，/]', m.group(1)):
        name = re.sub(r'[^A-Za-z]', '', part).lower().strip()
        if name and name not in pool:
            pool.append(name)
    return pool


def _extract_zebra_candidate(assistant_text: str, prompt_text: str) -> str | None:
    patterns = [
        r'house\s*1[^\.\n]{0,120}?\bis\s+([A-Za-z]+)',
        r'lives?\s+in\s+house\s*1[^\.\n]{0,120}?\bis\s+([A-Za-z]+)',
        r'final answer[^\.\n]{0,40}?\bis\s+([A-Za-z]+)',
        r'answer[^\.\n:]{0,20}?[:：]\s*([A-Za-z]+)',
    ]
    for pat in patterns:
        matches = re.findall(pat, assistant_text, flags=re.IGNORECASE)
        if matches:
            token = re.sub(r'[^A-Za-z]', '', matches[-1]).lower().strip()
            if token:
                return token

    names = _extract_name_pool(prompt_text)
    if not names:
        return None

    lower = assistant_text.lower()
    best_pos = -1
    best_name = None
    for name in names:
        for m in re.finditer(rf'\b{re.escape(name)}\b', lower):
            if m.start() >= best_pos:
                best_pos = m.start()
                best_name = name
    return best_name


def _extract_arc_candidate(assistant_text: str) -> str | None:
    matches = re.findall(r'(\d(?:[\s,]+\d){3,})', assistant_text)
    if not matches:
        return None
    cand = re.sub(r'[\s,]+', ' ', matches[-1]).strip()
    return cand if cand else None


def _extract_countdown_candidate(assistant_text: str) -> str | None:
    lines = [x.strip() for x in assistant_text.splitlines() if x.strip()]
    for line in reversed(lines):
        if '=' in line:
            left = line.split('=', 1)[0].strip()
            if re.fullmatch(r'[0-9+\-*/()\s]+', left) and any(op in left for op in '+-*/'):
                return re.sub(r'\s+', '', left)
        if re.fullmatch(r'[0-9+\-*/()\s]+', line) and any(op in line for op in '+-*/'):
            return re.sub(r'\s+', '', line)

    chunks = re.findall(r'([0-9+\-*/()\s]{5,})', assistant_text)
    for chunk in reversed(chunks):
        c = chunk.strip()
        if re.fullmatch(r'[0-9+\-*/()\s]+', c) and any(op in c for op in '+-*/'):
            return re.sub(r'\s+', '', c)
    return None


def _extract_math_candidate(assistant_text: str) -> str | None:
    ans = extract_math_answer(assistant_text[-800:])
    if ans is not None and str(ans).strip() != '':
        return str(ans).strip()

    patterns = [
        r'final answer[^\.\n]{0,60}?(-?\d+(?:/\d+)?(?:\.\d+)?)',
        r'answer[^\.\n:]{0,20}?[:：]\s*(-?\d+(?:/\d+)?(?:\.\d+)?)',
    ]
    for pat in patterns:
        matches = re.findall(pat, assistant_text, flags=re.IGNORECASE)
        if matches:
            return matches[-1].strip()
    return None


def _canonicalize_output(dataset: str, prompt_text: str, full_text: str) -> tuple[str, str | None, str | None, bool]:
    raw_parsed = _parse_answer(dataset, full_text)
    raw_clean = str(raw_parsed).strip() if raw_parsed is not None else None
    if raw_clean:
        return full_text, raw_clean, raw_clean, False

    assistant_text = _assistant_segment(full_text)

    candidate = _extract_boxed_anywhere(assistant_text)
    if not candidate:
        if dataset == 'zebra':
            candidate = _extract_zebra_candidate(assistant_text, prompt_text)
        elif dataset == 'arc':
            candidate = _extract_arc_candidate(assistant_text)
        elif dataset == 'countdown':
            candidate = _extract_countdown_candidate(assistant_text)
        else:
            candidate = _extract_math_candidate(assistant_text)

    if not candidate:
        return full_text, None, raw_clean, False

    canonical_text = full_text.rstrip() + f'\n\\boxed{{{candidate}}}'
    parsed = _parse_answer(dataset, canonical_text)
    parsed_clean = str(parsed).strip() if parsed is not None else None
    if not parsed_clean:
        return full_text, None, raw_clean, False
    return canonical_text, parsed_clean, raw_clean, True


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    sec_root = Path(args.sec_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(_data_path(sec_root, args.dataset, args.split))
    df = df[['data_source', 'prompt', 'reward_model', 'extra_info']].copy()
    df['difficulty'] = [_difficulty(ei, ds) for ei, ds in zip(df['extra_info'], df['data_source'])]

    buckets = sorted(df['difficulty'].dropna().astype(int).unique().tolist())
    sampled_parts = []
    for d in buckets:
        part = df[df['difficulty'] == d]
        if len(part) == 0:
            continue
        n = min(args.n_per_difficulty, len(part))
        sampled_parts.append(part.sample(n=n, random_state=args.seed + d))
    sample_df = pd.concat(sampled_parts, ignore_index=True) if sampled_parts else df.head(0)

    print(f'[INFO] dataset={args.dataset} split={args.split} total={len(df)} sampled={len(sample_df)} buckets={buckets}')

    has_cuda = torch.cuda.is_available()
    dtype = torch.float16 if has_cuda else torch.float32
    print(f'[INFO] loading model from {args.model_path}; cuda={has_cuda}; dtype={dtype}')
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=dtype,
        device_map='auto' if has_cuda else None,
        trust_remote_code=True,
    )
    model.eval()

    scorer_pass, scorer_reward = _scorers(args.dataset)

    rows: list[ProbeRow] = []
    for i, row in sample_df.reset_index(drop=True).iterrows():
        gt = row['reward_model'].get('ground_truth', '') if isinstance(row['reward_model'], dict) else ''
        target = extract_countdown_target(row['reward_model']) if args.dataset == 'countdown' else None

        msgs = _messages(row['prompt'])
        msgs = _apply_prompt_protocol(msgs, dataset=args.dataset, target=target)
        prompt_text = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer([prompt_text], return_tensors='pt')
        if has_cuda:
            inputs = {k: v.to(model.device) for k, v in inputs.items()}

        diff = int(row['difficulty'])

        for r in range(args.rollouts):
            gen_kwargs = dict(
                max_new_tokens=args.max_new_tokens,
                do_sample=(args.temperature > 0),
                top_p=args.top_p,
            )
            if args.temperature > 0:
                gen_kwargs['temperature'] = args.temperature

            with torch.no_grad():
                out_ids = model.generate(**inputs, **gen_kwargs)
            new_ids = out_ids[0][inputs['input_ids'].shape[-1]:]
            resp = tokenizer.decode(new_ids, skip_special_tokens=False)
            full_text = prompt_text + resp

            score_text, parsed, raw_parsed, canonicalized = _canonicalize_output(args.dataset, prompt_text, full_text)
            format_valid = int(parsed is not None and str(parsed).strip() != '')
            parse_success = format_valid
            pass_score = float(scorer_pass(solution_str=score_text, ground_truth=gt))
            reward_score = float(scorer_reward(solution_str=score_text, ground_truth=gt))

            cd_numbers_ok = None
            cd_target_ok = None
            if args.dataset == 'countdown' and parsed is not None and str(parsed).strip() != '':
                v = validate_countdown_expression(str(parsed), gt)
                cd_numbers_ok = int(bool(v.get('numbers_ok', False)))
                cd_target_ok = int(bool(v.get('target_ok', False)))

            rows.append(
                ProbeRow(
                    dataset=args.dataset,
                    difficulty=diff,
                    idx=int(i),
                    rollout_id=r,
                    format_valid=format_valid,
                    parse_success=parse_success,
                    pass_score=pass_score,
                    reward_score=reward_score,
                    parsed_answer=None if parsed is None else str(parsed),
                    raw_parsed_answer=None if raw_parsed is None else str(raw_parsed),
                    canonicalized=int(canonicalized),
                    ground_truth=gt,
                    response=resp,
                    countdown_numbers_ok=cd_numbers_ok,
                    countdown_target_ok=cd_target_ok,
                )
            )

        if (i + 1) % 10 == 0:
            print(f'[PROGRESS] {i+1}/{len(sample_df)} done')

    out_jsonl = out_dir / f'{args.dataset}_{args.split}_raw.jsonl'
    with out_jsonl.open('w', encoding='utf-8') as f:
        for r in rows:
            f.write(json.dumps(asdict(r), ensure_ascii=False) + '\n')

    result_df = pd.DataFrame([asdict(r) for r in rows])
    summary = result_df.groupby('difficulty').agg(
        n=('idx', 'count'),
        format_valid_rate=('format_valid', 'mean'),
        parse_success_rate=('parse_success', 'mean'),
        task_pass_rate=('pass_score', 'mean'),
        reward_mean=('reward_score', 'mean'),
        canonicalized_rate=('canonicalized', 'mean'),
    ).reset_index().sort_values('difficulty')

    out_csv = out_dir / f'{args.dataset}_{args.split}_summary.csv'
    summary.to_csv(out_csv, index=False)

    print('\n[SUMMARY]')
    print(summary.to_string(index=False))
    print(f'\n[RAW] {out_jsonl}')
    print(f'[SUMMARY_CSV] {out_csv}')


if __name__ == '__main__':
    main()
