# Scheduler Test Plan

## Scope
This plan validates the scheduler core behavior before full RL training:
- UCB sampling probability behavior
- with-replacement sampling
- decayed sample state update
- re-bucketing guardrails and migration direction rules

## Unit Tests
Run:
```bash
pytest tests/test_scheduler_unit.py -q
```

Covered checks:
1. `probability_normalization`
2. `probability_floor`
3. `with_replacement_sampling`
4. `decayed_state_update`
5. `warmup_no_migration`
6. `min_obs_guard`
7. `neighbor_only_migration`

Expected outcomes:
- All cluster probabilities are in `[0, 1]` and sum to `1`.
- A very low-UCB cluster still keeps non-zero probability after floor mixing.
- Sampling repeats sample IDs and does not shrink cluster members.
- Decayed state update follows `s <- decay * s + (1 - decay) * |A|` exactly.
- No migration occurs while `global_step < warmup_steps`.
- No migration occurs when `obs_count < min_obs_for_rebucket`.
- Migrations only move to adjacent easier cluster (`hard -> medium`, `medium -> easy`).

## Toy Simulation
Run:
```bash
python scripts/toy_simulation_rebucket.py
```

Outputs:
- `artifacts/toy_rebucket_simulation.png`
- `artifacts/toy_rebucket_summary.json`

Scenarios and expected outcomes:
1. **Scenario A (normal drifting):** medium cluster drifts as a whole.
   - Expected: no migration.
2. **Scenario B (persistent high outlier):** one medium sample remains above threshold for consecutive refreshes.
   - Expected: trigger accumulation then migration `medium -> easy` after `MIGRATION_CONSECUTIVE`.
3. **Scenario C (single-step spike):** one spike appears only once.
   - Expected: no migration, trigger streak resets.

## Smoke Config (Future Integration)
Config file:
- `configs/scheduler_smoke_test.yaml`

Purpose:
- very small integration setup (small sample pool, short steps, full logging)
- enables with-replacement and re-bucketing for quick end-to-end scheduler smoke
