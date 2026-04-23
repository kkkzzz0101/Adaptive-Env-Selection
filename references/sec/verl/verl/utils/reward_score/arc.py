import re
import json
import random
from typing import Any

from verl.utils.reward_score.utils import extract_solution


def _flatten_to_tokens(obj: Any) -> list[str]:
    if obj is None:
        return []

    if isinstance(obj, dict):
        # Prefer explicit ARC target field when available.
        if 'target' in obj:
            return _flatten_to_tokens(obj['target'])
        if 'ground_truth' in obj:
            return _flatten_to_tokens(obj['ground_truth'])

        toks: list[str] = []
        for v in obj.values():
            toks.extend(_flatten_to_tokens(v))
        return toks

    if isinstance(obj, (list, tuple)):
        toks: list[str] = []
        for v in obj:
            toks.extend(_flatten_to_tokens(v))
        return toks

    s = str(obj)
    nums = re.findall(r'-?\d+', s)
    if nums:
        return nums

    return s.strip().lower().split()


def normalize_arc_sequence(obj: Any) -> str:
    toks = _flatten_to_tokens(obj)
    if not toks:
        return ''
    return ' '.join(toks).strip().lower()


def evaluate_equation(answer: Any, target: Any) -> bool:
    ans_norm = normalize_arc_sequence(answer)
    tgt_norm = normalize_arc_sequence(target)

    if ans_norm == '' or tgt_norm == '':
        return False
    return ans_norm == tgt_norm


def compute_score(solution_str, ground_truth, method='strict', format_score=0.1, score=1.0):
    """Scoring for ARC 1D-style output matching.

    For ARC we treat both model answer and target as token sequences and compare
    after normalization into a single space-delimited digit string.
    """
    gt_obj = ground_truth
    if isinstance(gt_obj, str):
        try:
            gt_obj = json.loads(gt_obj)
        except Exception:
            # Keep raw string fallback.
            pass

    if isinstance(gt_obj, dict):
        target = gt_obj.get('target', '')
    else:
        target = gt_obj

    answer = extract_solution(solution_str=solution_str)
    do_print = random.randint(1, 64) == 1

    if do_print:
        print('--------------------------------')
        print(f'Target(raw): {target}')
        print(f'Extracted answer(raw): {answer}')
        print(f'Target(norm): {normalize_arc_sequence(target)}')
        print(f'Answer(norm): {normalize_arc_sequence(answer)}')
        print(f'Solution string: {solution_str}')

    if answer is None:
        if do_print:
            print('No answer found')
        return 0

    try:
        ok = evaluate_equation(answer, target)
        if ok:
            if do_print:
                print('Correct answer')
            return score
        if do_print:
            print('Wrong answer')
        return format_score
    except Exception:
        if do_print:
            print('Error evaluating answer')
        return format_score


def get_arc_compute_score(correct_score, format_score):
    return lambda solution_str, ground_truth: compute_score(
        solution_str,
        ground_truth,
        method='strict',
        format_score=format_score,
        score=correct_score,
    )
