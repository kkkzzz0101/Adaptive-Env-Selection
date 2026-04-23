# AES Progress (SEC + DUMP Random Baseline)

## What We Have Done
- Unified prompts for 4 datasets (Countdown / ARC-1D / Zebra / Math), enforcing final `\\boxed{...}` protocol.
- Fixed parsing and target alignment in SEC-style data handling:
  - Countdown target uses ground_truth object (numbers, target) and expression-based checking.
  - ARC ground truth comparison is flattened numeric sequence vs parsed boxed output.
- Added smoke debug print in DUMP training loop to inspect:
  - problem (prompt)
  - model response
  - reward
  - group advantage

## Current Blocker (Resolved)
- DUMP reward routing previously did not recognize `arc_train`, `zebra_train`, `countdown_train`, `math_train`.
- This caused fallback `score=0.0` and degenerate zero advantages.
- Fixed by normalizing data_source (`*_train/*_val/*_test`) and routing to the proper scorer.

## Validation Snapshot
- Smoke run log: `logs/baseline_random_dump_smoke_routefix.log`
- `Unknown data_source` count: **0**
- This confirms data-source routing now matches current SEC mixed dataset naming.

## Next Task
- Run full random baseline with current DUMP pipeline (no scheduler), then collect:
  - per-dataset/difficulty accuracy
  - reward/advantage diagnostics from smoke debug samples
  - checkpointed validation trend

## Status
- DONE: Prompt + parsing protocol fixed
- DONE: Smoke debug logging added
- DONE: DUMP scorer routing fix (`*_train/_val/_test` normalized)
- TODO: Full random baseline rerun and metrics report

## Repo Note
- `.gitignore` now uses a whitelist for `references/DUMP/` so only AES hotfix files are tracked.
- Tracked DUMP files: `verl/trainer/ppo/ray_trainer.py`, `verl/utils/reward_score/__init__.py`, `verl/utils/reward_score/sec4_mixed.py`.
- The nested upstream git metadata was preserved as `references/DUMP/.git.upstream.bak`.
