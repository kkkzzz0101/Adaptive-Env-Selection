# Figure Plan

| ID | Output | Type | Purpose | Data source |
|---|---|---|---|---|
| Fig. 1 | `fig_method_pipeline` | Pipeline diagram | Show AES feedback loop: Math+Zebra tasks, initial buckets, UCB sampling, GRPO training, advantage-state update, and micro-bucket re-bucketing. | Method implementation in `src/scheduler/adaptive_curriculum_scheduler.py` |
| Fig. 2 | `fig_step200_math_zebra` | Grouped bar chart | Summarize the clearest positive result: scheduler without re-bucketing beats random baseline on Math and Zebra at step 200. | `experiments/results/math_zebra_2data/baseline_vs_norebucket_metrics.csv` |
| Fig. 3 | `fig_ucb_score_drift` | Line plot | Show UCB score drift across clusters: C0/C1 dominate early, while C3/C4 dominate later. | User-provided UCB trace |
| Fig. 4 | `fig_initial_accuracy_profile` | Line plot | Show the initial accuracy test that diagnoses why label-only initialization is coarse. | User-provided initial accuracy test values |
| Fig. 5 | `fig_rebucket_composition` | Stacked bar chart | Visualize Experiment 1 task-composition drift under difficulty-label initialization, using both micro-bucket and sample counts. | `docs/result_report.md` composition table |
| Fig. 6 | `fig_inferred_transition_matrix` | Transition matrix | Show the inferred adjacent, lower-index transition structure from aggregate micro-bucket deltas. | `docs/result_report.md` composition table |
| Fig. 7 | `fig_toy_rebucket_guardrails` | Multi-panel line plot | Show that re-bucketing responds to persistent outliers rather than normal drift or one-step spikes. | `artifacts/toy_rebucket_summary.json` |

All generated figures are saved as PDF and PNG under `paper/figures/`.
