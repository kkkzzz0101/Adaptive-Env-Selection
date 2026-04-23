import ast
import json
import re
from collections import Counter
from typing import Any, Dict, List, Optional


def _to_ground_truth_obj(ground_truth: Any) -> Any:
    if isinstance(ground_truth, str):
        s = ground_truth.strip()
        if s.startswith('{') and s.endswith('}'):
            try:
                return json.loads(s)
            except Exception:
                return ground_truth
    return ground_truth


def _extract_assistant_text(solution_str: str) -> str:
    if not isinstance(solution_str, str):
        return str(solution_str)
    if "Assistant:" in solution_str:
        return solution_str.split("Assistant:", 1)[1]
    if "<|im_start|>assistant" in solution_str:
        return solution_str.split("<|im_start|>assistant", 1)[1]
    return solution_str


def _extract_boxed(text: str) -> Optional[str]:
    matches = re.findall(r"\\boxed\{([^{}]*)\}", text)
    if matches:
        return matches[-1].strip()
    return None


def _safe_eval_arith(expr: str) -> Optional[float]:
    # Allow only basic arithmetic expressions.
    expr = expr.strip()
    if not expr:
        return None
    if re.search(r"[^0-9\s\+\-\*/\(\)\.]+", expr):
        return None
    try:
        node = ast.parse(expr, mode='eval')
    except Exception:
        return None

    allowed = (
        ast.Expression, ast.BinOp, ast.UnaryOp, ast.Constant,
        ast.Add, ast.Sub, ast.Mult, ast.Div, ast.UAdd, ast.USub,
        ast.Load, ast.Pow, ast.Mod, ast.FloorDiv,
    )
    for n in ast.walk(node):
        if not isinstance(n, allowed):
            return None
        if isinstance(n, ast.Constant) and not isinstance(n.value, (int, float)):
            return None

    try:
        val = eval(compile(node, '<expr>', 'eval'), {'__builtins__': {}}, {})
    except Exception:
        return None
    try:
        return float(val)
    except Exception:
        return None


def _extract_int_list(s: str) -> List[int]:
    return [int(x) for x in re.findall(r"-?\d+", s)]


def _extract_uint_list(s: str) -> List[int]:
    return [int(x) for x in re.findall(r"\d+", s)]


def _score_countdown(solution_str: str, gt_obj: Dict[str, Any]) -> float:
    numbers_raw = str(gt_obj.get('numbers', ''))
    target_raw = gt_obj.get('target', None)
    if target_raw is None:
        return 0.0

    gt_numbers = [int(x.strip()) for x in numbers_raw.split(',') if x.strip()]
    try:
        target = float(target_raw)
    except Exception:
        return 0.0

    text = _extract_assistant_text(solution_str)
    expr = _extract_boxed(text) or text
    expr = expr.strip()

    used_numbers = _extract_uint_list(expr)
    if Counter(used_numbers) != Counter(gt_numbers):
        return 0.0

    val = _safe_eval_arith(expr)
    if val is None:
        return 0.0

    return 1.0 if abs(val - target) < 1e-6 else 0.0


def _score_zebra(solution_str: str, gt_obj: Dict[str, Any]) -> float:
    target = str(gt_obj.get('target', '')).strip().lower()
    if not target:
        return 0.0

    text = _extract_assistant_text(solution_str)
    cand = _extract_boxed(text)
    if cand is None:
        # fallback: use last alphabetical token
        toks = re.findall(r"[A-Za-z]+", text)
        cand = toks[-1] if toks else ''
    cand = cand.strip().lower()

    return 1.0 if cand == target else 0.0


def _score_arc1d(solution_str: str, gt_obj: Dict[str, Any]) -> float:
    target = str(gt_obj.get('target', '')).strip()
    if not target:
        return 0.0
    gt_nums = _extract_int_list(target)
    if not gt_nums:
        return 0.0

    text = _extract_assistant_text(solution_str)
    cand = _extract_boxed(text) or text
    pred_nums = _extract_int_list(cand)

    return 1.0 if pred_nums == gt_nums else 0.0


def _normalize_sec4_data_source(data_source: str) -> str:
    ds = str(data_source or '')
    for suffix in ('_train', '_val', '_test'):
        if ds.endswith(suffix):
            ds = ds[:-len(suffix)]
            break

    # Accept historical aliases.
    if ds == 'arc':
        return 'arc1d'
    return ds


def compute_score(data_source: str, solution_str: str, ground_truth: Any) -> float:
    gt_obj = _to_ground_truth_obj(ground_truth)
    if isinstance(gt_obj, str):
        # Unexpected for sec4 tasks, return 0 conservatively.
        return 0.0

    ds = _normalize_sec4_data_source(data_source)

    if ds.startswith('countdown_diff_') or ds == 'countdown':
        return _score_countdown(solution_str, gt_obj)
    if ds.startswith('zebra_diff_') or ds == 'zebra':
        return _score_zebra(solution_str, gt_obj)
    if ds.startswith('arc1d_diff_') or ds == 'arc1d':
        return _score_arc1d(solution_str, gt_obj)
    return 0.0
