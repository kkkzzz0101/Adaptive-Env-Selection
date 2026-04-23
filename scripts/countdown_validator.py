from __future__ import annotations

import json
import re
from collections import Counter
from typing import Any


_ALLOWED_EXPR_PATTERN = re.compile(r"^[0-9+\-*/()\s.]+$")


def normalize_ground_truth(ground_truth: Any) -> tuple[list[int] | None, float | None]:
    """Return (numbers, target) parsed from SEC countdown ground_truth."""
    gt = ground_truth
    if isinstance(gt, str):
        try:
            gt = json.loads(gt)
        except Exception:
            return None, None

    if not isinstance(gt, dict):
        return None, None

    numbers = gt.get("numbers")
    target = gt.get("target")

    if isinstance(numbers, str):
        try:
            nums = [int(x.strip()) for x in numbers.split(",") if x.strip() != ""]
        except Exception:
            return None, None
    elif isinstance(numbers, list):
        try:
            nums = [int(x) for x in numbers]
        except Exception:
            return None, None
    else:
        return None, None

    try:
        tgt = float(target)
    except Exception:
        return None, None

    return nums, tgt


def extract_numbers_from_expression(expression: str) -> list[int]:
    return [int(x) for x in re.findall(r"\d+", expression)]


def numbers_used_exactly_once(expression: str, available_numbers: list[int]) -> bool:
    used = extract_numbers_from_expression(expression)
    return Counter(used) == Counter(available_numbers)


def evaluate_expression(expression: str) -> float | None:
    if not isinstance(expression, str):
        return None
    expr = expression.strip()
    if not expr:
        return None
    if not _ALLOWED_EXPR_PATTERN.fullmatch(expr):
        return None
    try:
        return float(eval(expr, {"__builtins__": None}, {}))
    except Exception:
        return None


def evaluate_against_target(expression: str, target: float, tol: float = 1e-6) -> bool:
    value = evaluate_expression(expression)
    if value is None:
        return False
    return abs(value - target) <= tol


def validate_countdown_expression(expression: str, ground_truth: Any, tol: float = 1e-6) -> dict[str, Any]:
    """Validate expression with countdown rules.

    Returns dict with:
      - numbers_ok: bool
      - target_ok: bool
      - value: float | None
      - numbers: list[int] | None
      - target: float | None
      - is_valid: bool
    """
    numbers, target = normalize_ground_truth(ground_truth)
    if numbers is None or target is None:
        return {
            "numbers_ok": False,
            "target_ok": False,
            "value": None,
            "numbers": numbers,
            "target": target,
            "is_valid": False,
            "error": "bad_ground_truth",
        }

    numbers_ok = numbers_used_exactly_once(expression, numbers)
    value = evaluate_expression(expression)
    target_ok = (value is not None) and (abs(value - target) <= tol)

    return {
        "numbers_ok": bool(numbers_ok),
        "target_ok": bool(target_ok),
        "value": value,
        "numbers": numbers,
        "target": target,
        "is_valid": bool(numbers_ok and target_ok),
    }
