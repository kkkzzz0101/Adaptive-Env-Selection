from __future__ import annotations

import re
from typing import Any

PROTOCOL_MARKER = "### Output Protocol (AES v3)"


def infer_dataset_from_data_source(data_source: Any) -> str | None:
    text = str(data_source).lower()
    if text.startswith("countdown"):
        return "countdown"
    if text.startswith("zebra"):
        return "zebra"
    if text.startswith("arc"):
        return "arc"
    if text.startswith("math"):
        return "math"
    return None


def extract_countdown_target(reward_model: Any) -> str | None:
    if isinstance(reward_model, dict):
        gt = reward_model.get("ground_truth")
    else:
        gt = None

    if gt is None:
        return None

    if isinstance(gt, dict):
        target = gt.get("target")
        if isinstance(target, int):
            return str(target)
        if isinstance(target, float) and target.is_integer():
            return str(int(target))
        if isinstance(target, str) and re.fullmatch(r"-?\d+", target.strip()):
            return target.strip()
        return None

    if isinstance(gt, (int, float)):
        if int(gt) == gt:
            return str(int(gt))
        return None

    s = str(gt).strip()
    if re.fullmatch(r"-?\d+", s):
        return s
    return None


def build_protocol_suffix(dataset: str | None, target: str | None = None) -> str:
    lines: list[str] = [
        PROTOCOL_MARKER,
        "- Keep the entire response under 1024 tokens.",
        "- The final line must be exactly one boxed answer: \\boxed{...}.",
        "- Do not add any text after the final boxed answer.",
    ]

    if dataset == "arc":
        lines.extend(
            [
                "- Do not provide reasoning.",
                "- Output exactly one line only: \\boxed{...}.",
                "- Inside \\boxed{...}, output only digits separated by single spaces in one line (row-major order).",
                "- Do not use LaTeX array/tabular/alignment/newlines inside the box.",
            ]
        )
    elif dataset == "countdown":
        lines.extend(
            [
                "- Do not provide reasoning.",
                "- Output exactly one line only: \\boxed{...}.",
                "- Inside \\boxed{...}, output only one ASCII arithmetic expression using numbers and + - * / ( ).",
                "- Do not include '=' or any words inside the box.",
            ]
        )
        if target is not None:
            lines.append(f"- Ensure the expression evaluates exactly to target {target}.")
    elif dataset == "zebra":
        lines.extend(
            [
                "- Use at most 2-3 short reasoning sentences.",
                "- Inside \\boxed{...}, output only the final person's name.",
            ]
        )
    else:
        lines.append("- Use at most 2-3 short reasoning sentences.")

    return "\n".join(lines)


def append_protocol_text(content: str, dataset: str | None, target: str | None = None) -> str:
    if "### Output Protocol (AES v" in content:
        return content

    suffix = build_protocol_suffix(dataset=dataset, target=target)
    return content.rstrip() + "\n\n" + suffix
