# DURCL for RL-Based LLM Post-Training

Course project report  
Date: 2026-04-28

## Abstract

Reinforcement-learning post-training for large language models usually mixes multiple task distributions. This project focuses on a minimal but heterogeneous Math+Zebra setting: mathematical problem solving and logic puzzles. Uniform or random sampling ignores that different distributions can become more or less useful as the model learns. This project studies DURCL, Automated Distribution-Level Rebucketing Curriculum Learning, an online curriculum mechanism that groups training examples into difficulty clusters, samples clusters with an Upper Confidence Bound (UCB)-style policy, and periodically refines cluster membership through micro-bucket rebucketing.

The current implementation integrates Math+Zebra reasoning tasks with a DUMP/verl training path and a standalone DURCL scheduler. The strongest current result is not a universal downstream accuracy claim. Rather, the evidence shows that DURCL produces meaningful and interpretable redistribution signals: a no-rebucket scheduler improves Math and Zebra validation accuracy at step 200 over a random baseline, while rebucketing changes cluster composition in structured ways. Under accuracy-based initialization, the available no-rebucket validation result reaches 0.500 on Math and 0.300 on Zebra at step 200; the corresponding rebucketing run is still pending. The main conclusion is therefore that DURCL is a promising learner-state-aware environment selection mechanism, but stricter multi-seed evaluation is still needed.

## 1. Introduction

Modern RL-based LLM post-training often combines heterogeneous environments. In this report, the training mixture is narrowed to Math and Zebra. Math tests mathematical problem solving, while Zebra tests logic-puzzle reasoning. A random sampler treats all examples as exchangeable. This is simple, but it can waste compute on examples that are either already solved, too noisy, or not currently aligned with the model's learning frontier.

The issue is especially visible in GRPO-style training. Since updates are driven by group-relative advantages, examples that are too easy or too hard can both provide little useful signal: if all sampled rollouts succeed or all fail, the within-group advantage is close to uninformative. The best training distribution is therefore not fixed. DURCL addresses this by making data selection adaptive. It asks: can we use online learning signals from training, especially advantage magnitude, to decide which environment clusters should receive more sampling probability and whether examples should move between difficulty buckets?

This is close in spirit to curriculum learning and distribution-level curriculum scheduling. Curriculum learning argues that example order can change optimization behavior, while DUMP frames RL post-training curriculum selection as a distribution-level scheduling problem using advantage magnitude and UCB. DURCL keeps this high-level motivation but models the mixed Math+Zebra curriculum as a non-stationary bandit: each cluster is an arm whose usefulness can drift as the model learns.

## 2. Problem Setting

The project studies RL-based post-training over a Math+Zebra reasoning dataset.

Training tasks:

| Task | Target behavior | Reward/scoring style |
|---|---|---|
| Zebra | Solve a logic puzzle and output the target entity | Parse final boxed answer or fallback token |
| Math | Solve math problems | Routed to existing math scorer |

The data builder constructs fixed task/difficulty buckets: Zebra uses levels 1-4, while Math uses levels 1-5. This asymmetry matters because cluster C4 is Math-only from the beginning. The repository uses Qwen2.5-1.5B-Instruct as the example model path in the launch scripts, DUMP/verl as the training core, and GRPO as the configured advantage estimator in the visible training launchers. The explicit random-baseline scripts set `data.enable_curriculum_learning=False`, which makes them appropriate baselines for comparing against scheduler-enabled variants.

The two tasks also have different initial accuracy scales. This makes Math+Zebra a compact setting for testing whether a scheduler can respond to learner-state drift rather than only following nominal difficulty labels.

## 3. Method

DURCL has two layers: cluster-level adaptive sampling and micro-bucket rebucketing.

Compared with the original DUMP algorithm, DURCL makes two main method changes. DUMP stores a distribution-level sliding window `A_w[d_j]` of recent absolute advantages and keeps the last `k` values for each distribution. DURCL instead uses an exponential-decay advantage state, which reduces retained history and makes updates smoother. The second change is micro-bucket rebucketing. In Math+Zebra data, the scoring rules, response formats, problem types, and learning rates differ; a fixed cluster structure can become stale even if UCB weights change. DURCL therefore allows persistent structural mismatches to be corrected between neighboring clusters, so the curriculum structure can adapt as well as the sampling weights.

### 3.1 Initial Clustering

Examples are assigned to ordered difficulty clusters. The scheduler supports explicit `cluster_id` or `difficulty_band` annotations. It can also build clusters from a calibration map where higher accuracy means an easier initial cluster. The current reported experiments mainly use coarse difficulty-based initialization, which is intentionally treated as imperfect.

The implementation convention is that cluster `0` is easiest and larger cluster IDs are harder. Some docs show a default three-cluster configuration, while the Math+Zebra analysis uses five reported clusters (`cluster_0` through `cluster_4`).

### 3.2 Online State Tracking

For each sample, DURCL maintains a decayed state:

```text
s_i <- decay * s_i + (1 - decay) * |A_i|
```

where `|A_i|` is the absolute advantage signal observed for that sample. This state is used as an online proxy for learner interaction with the example. Cluster state is computed from active samples within a recent window, giving each cluster an active mean and standard deviation.

### 3.3 UCB Cluster Sampling

For each cluster, DURCL computes a UCB-style score:

```text
ucb_c = value_c + beta * sqrt(2 * log(total_draws + 1) / (sample_count_c + 1))
```

The scores are converted to sampling probabilities through softmax and mixed with a probability floor. This gives the scheduler both exploitation and exploration: clusters with high current value receive more probability, while under-sampled clusters retain a nonzero chance of being selected.

Sampling is with replacement. This is important for RL post-training because a batch draw should not remove a prompt from later training.

### 3.4 Micro-bucket Rebucketing

The scheduler assumes each curriculum cluster is internally coherent: examples inside the same cluster should have similar current learning dynamics. This can fail when initial clusters are built only from coarse difficulty labels or noisy pre-evaluation signals. DURCL therefore splits each initial task--cluster group into micro-buckets. A micro-bucket is the smallest unit that can move; individual samples are never moved independently. This reduces variance and makes rebucketing a structural correction rather than instance-level filtering.

The design separates scheduling from structural correction. UCB decides which cluster should be sampled now. Rebucketing is a slower monitoring layer that asks whether a small group inside a cluster consistently looks more similar to a neighboring cluster. This prevents the rebucketing module from replacing the scheduler with unconstrained reclustering.

During training, each micro-bucket records recent mean absolute advantage. Given a recent observation window, its dynamics signature is

```text
z_g = (level_g, slope_g)
```

where `level_g` is recent average `|A|` and `slope_g` is the fitted slope of the advantage trajectory. Each cluster also has a dynamics center, computed from the micro-buckets currently assigned to it. After standardizing level and slope, each micro-bucket is compared only with its current cluster and adjacent clusters:

```text
j*(g) = argmin_{j in {k-1, k, k+1}} || z_g - z_j ||^2
```

If the same neighboring cluster is consistently closer across multiple rebucketing checks, the micro-bucket becomes a correction candidate. The main operation is an adjacent mutual swap: if a micro-bucket in `C_k` is closer to `C_{k+1}` and a micro-bucket in `C_{k+1}` is closer to `C_k`, DURCL swaps them. This keeps cluster sizes stable and prevents rebucketing from replacing the scheduler. The scheduler remains responsible for curriculum progression; rebucketing only corrects local structural mismatches.

The important scientific role of rebucketing is not just to improve immediate accuracy. It is a diagnostic for whether the initial difficulty buckets are aligned with the model's actual learning state.

## 4. Implementation Summary

Key implementation components:

| Component | Repository evidence |
|---|---|
| Scheduler core | `src/scheduler/adaptive_curriculum_scheduler.py` |
| Scheduler config | `configs/scheduler/default.yaml` |
| Unit tests | `tests/test_scheduler_unit.py` |
| Toy rebucketing simulation | `scripts/toy_simulation_rebucket.py`, `artifacts/toy_rebucket_summary.json` |
| DUMP random baseline launch | `scripts/run_baseline_random_dump_2gpu.sh` |
| Main result summary | `docs/result_report.md` |

The unit tests cover probability normalization, probability floor behavior, with-replacement sampling, decayed-state updates, warmup guards, minimum-observation guards, and neighbor-only migration. The toy simulation checks three cases: normal cluster drift does not trigger migration, a persistent high outlier migrates after repeated refreshes, and a single-step spike does not migrate.

One reproducibility caveat is that some documentation references scripts that are not present in the current tree, such as `scripts/smoke_scheduler.py`, `experiments/baselines/run_scheduler_smoke.sh`, and `experiments/baselines/run_random_mixed_formal_4gpu_qwen15b.sh`. Runtime artifacts such as checkpoints, logs, and outputs are also intentionally git-ignored. As a result, this report treats checked-in summaries and CSV files as the evidence base, rather than claiming to reproduce the full training runs locally.

## 5. Experiments

### 5.1 Math+Zebra Step-200 Validation

The first reported comparison focuses on Math and Zebra. It compares random sampling to the scheduler without rebucketing at step 200.

| Run | Step | Math train | Zebra train |
|---|---:|---:|---:|
| Random baseline | 100 | 0.440 | 0.263 |
| Random baseline | 200 | 0.460 | 0.250 |
| Scheduler, no re-bucket | 60 | 0.440 | 0.250 |
| Scheduler, no re-bucket | 100 | 0.420 | 0.237 |
| Scheduler, no re-bucket | 200 | 0.480 | 0.338 |

At step 200, the no-rebucket scheduler improves over random sampling by `+0.020` on Math and `+0.088` on Zebra. This is the clearest positive downstream result in the current repo.

### 5.2 UCB Score Drift Diagnostic

The UCB trace provides a mechanism-level check that the sampler is active, not fixed. The values below are UCB scores rather than normalized sampling probabilities:

| Step | C0 | C1 | C2 | C3 | C4 |
|---:|---:|---:|---:|---:|---:|
| 60 | 0.329 | 0.301 | 0.259 | 0.238 | 0.230 |
| 80 | 0.250 | 0.264 | 0.243 | 0.252 | 0.235 |
| 100 | 0.190 | 0.210 | 0.250 | 0.256 | 0.220 |
| 120 | 0.135 | 0.189 | 0.204 | 0.275 | 0.216 |
| 140 | 0.072 | 0.148 | 0.176 | 0.275 | 0.257 |
| 160 | 0.101 | 0.141 | 0.184 | 0.233 | 0.255 |
| 180 | 0.034 | 0.106 | 0.197 | 0.225 | 0.264 |
| 200 | 0.081 | 0.101 | 0.182 | 0.295 | 0.253 |

The main pattern is a ranking reversal. Early in training, C0 and C1 have the largest scores. By step 200, C3 and C4 dominate. This supports the claim that UCB is effective as a dynamic scheduler: it responds to training signals instead of preserving the initial difficulty prior. It also reveals a drift problem: sampling pressure can move substantially across clusters, so the report must monitor cluster and task composition rather than only final accuracy.

### 5.3 Re-Bucketing Runs

The difficulty-init re-bucket report gives the following validation results:

| Run | Validation point | Math train | Zebra train |
|---|---|---:|---:|
| Random baseline | step 200 | 0.460 | 0.250 |
| Scheduler, no re-bucket | step 200 | 0.480 | 0.338 |
| Re-bucket | step 200 | 0.540 | 0.287 |
| Re-bucket 100 -> 300 | step 300 | 0.520 | 0.287 |
| Baseline 200 -> 300 | final | 0.560 | 0.275 |

This result should be interpreted carefully. Rebucketing has a strong Math score at step 200 and remains better than the baseline on Zebra at step 300. However, the baseline final Math score is higher than the re-bucket 100 -> 300 run. Therefore, rebucketing currently supports a structural-correction claim more strongly than a final-accuracy claim.

### 5.4 Experiment 1: Difficulty-Label Initialization and Composition Drift

This experiment studies whether online rebucketing can correct imperfect initial curriculum buckets. It uses no accuracy-based calibration: initial cluster assignment is based only on coarse dataset difficulty labels. This is intentionally noisy, and it tests whether rebucketing can still learn structural corrections when initialization is rough.

The initial accuracy test already shows why the label-only initialization is coarse:

| Task | d1 | d2 | d3 | d4 | d5 |
|---|---:|---:|---:|---:|---:|
| Zebra | 0.30 | 0.20 | 0.25 | 0.15 | - |
| Math | 0.70 | 0.60 | 0.35 | 0.35 | 0.10 |

Two details matter. First, Zebra is not strictly monotone by nominal difficulty because d3 is higher than d2. Second, Math and Zebra live on different initial-accuracy scales. This is why a pure label-based bucket assignment is a coarse proxy for learner competence. Even under this rough initialization, rebucketing discovers meaningful redistribution patterns; accuracy-based initialization is the next step to test whether better initial buckets lead to cleaner and more useful corrections.

At re-bucket step 300, reported cluster statistics were:

| Cluster | A_mean | Size | Migration in | Migration out |
|---|---:|---:|---:|---:|
| C0 | 0.152 | 248 | 6 | 2 |
| C1 | 0.137 | 152 | 7 | 9 |
| C2 | 0.160 | 196 | 8 | 7 |
| C3 | 0.211 | 140 | 4 | 6 |
| C4 | 0.178 | 64 | 1 | 2 |

The composition analysis shows that rebucketing changes task mix, not only within-task difficulty.

Initial composition by micro-bucket count:

| Cluster | Math micro-buckets | Zebra micro-buckets | Total |
|---|---:|---:|---:|
| C0 | 5 | 5 | 10 |
| C1 | 5 | 5 | 10 |
| C2 | 5 | 5 | 10 |
| C3 | 5 | 5 | 10 |
| C4 | 5 | 0 | 5 |

Initial composition by sample count:

| Cluster | Math samples | Zebra samples | Total |
|---|---:|---:|---:|
| C0 | 80 | 100 | 180 |
| C1 | 80 | 100 | 180 |
| C2 | 80 | 100 | 180 |
| C3 | 80 | 100 | 180 |
| C4 | 80 | 0 | 80 |

Note that C4 starts as Math-only because Zebra has four difficulty clusters while Math has five.

Final composition by micro-bucket count:

| Cluster | Math micro-buckets | Zebra micro-buckets | Total |
|---|---:|---:|---:|
| C0 | 8 | 6 | 14 |
| C1 | 2 | 6 | 8 |
| C2 | 6 | 5 | 11 |
| C3 | 5 | 3 | 8 |
| C4 | 4 | 0 | 4 |

Final composition by sample count:

| Cluster | Initial Math | Initial Zebra | Final Math | Final Zebra |
|---|---:|---:|---:|---:|
| C0 | 80 | 100 | 128 | 120 |
| C1 | 80 | 100 | 32 | 120 |
| C2 | 80 | 100 | 96 | 100 |
| C3 | 80 | 100 | 80 | 60 |
| C4 | 80 | 0 | 64 | 0 |

The main drift patterns are interpretable: C0 grows mainly through Math inflow, C1 becomes Zebra-heavy, C3 loses Zebra, and C4 remains Math-only while shrinking by one Math micro-bucket. This is useful evidence that the migration rule is not random. Math C0 is especially interpretable: it is expanded, and Math d1 is also the highest initial-accuracy group. This suggests that the scheduler is recovering some learner-aligned grouping even from crude labels.

The checked-in summary does not include raw migration-event logs, so a transition matrix should be described as an inferred net transition from aggregate micro-bucket deltas. A minimal adjacent-left-flow decomposition gives the following interpretation:

| Task | Inferred adjacent moves |
|---|---|
| Math | C1 -> C0: 3, C3 -> C2: 1, C4 -> C3: 1 |
| Zebra | C1 -> C0: 1, C2 -> C1: 2, C3 -> C2: 2 |

This supports the transition-matrix story: migration is not a random jump pattern; it is mostly adjacent and toward earlier/easier buckets. It also reveals a limitation: task composition can drift as a side effect of rebucketing, so a future version should constrain or explicitly measure task balance.

### 5.5 Experiment 2: Accuracy-Based Initialization

The second experiment repeats the 200-step comparison under accuracy-based initialization. It compares no-rebucket and rebucket variants after the initial buckets are built from accuracy rather than raw difficulty labels. This is the direct test of the next hypothesis from Experiment 1: better initial buckets should lead to cleaner and more useful corrections. The no-rebucket validation result is now available; the corresponding rebucketing run is still pending.

| Run | Step | Math | Zebra |
|---|---:|---:|---:|
| Acc-init, no re-bucket | 100 | 0.480 | 0.263 |
| Acc-init, no re-bucket | 200 | 0.500 | 0.300 |
| Acc-init, re-bucket | 200 | pending | pending |

The available no-rebucket result improves over the random step-200 reference of 0.460 on Math and 0.250 on Zebra. This supports the interpretation that accuracy-based initialization is cleaner than raw difficulty labels. The missing rebucketing run remains the key test of whether micro-bucket correction adds value after a better initialization.

### 5.6 Toy Re-Bucketing Simulation

The toy simulation validates the intended guardrails:

| Scenario | Expected behavior | Observed migration |
|---|---|---|
| Normal drifting medium cluster | No migration | None |
| Persistent high outlier | Migrate after repeated triggers | Step 300 |
| Single-step spike | No migration | None |

This supports the mechanism-level correctness of the rebucketing rule: it responds to persistent outliers rather than one-off spikes.

## 6. Discussion

The evidence supports three claims.

First, DURCL can produce useful adaptive sampling behavior. The no-rebucket scheduler improves both Math and Zebra at step 200 in the two-task experiment, and the accuracy-initialized no-rebucket run improves to 0.500 on Math and 0.300 on Zebra. The cluster probability trajectory also changes over training, showing that the scheduler is not behaving like a fixed sampler.

Second, rebucketing produces meaningful structural corrections. The cluster migration and composition tables show coherent redistribution patterns. This matters because the initial buckets were only difficulty-based, not learner-state-based. Rebucketing is therefore partially recovering a grouping that better reflects the observed training signal.

Third, the current downstream performance story should stay conservative. The no-rebucket scheduler improves both Math and Zebra at step 200, but the rebucketing result is stronger as structural evidence than as a final-accuracy result, and the accuracy-initialized rebucketing run is still missing. The correct project claim is not "DURCL wins overall." The correct claim is: DURCL exposes and partially corrects mismatch between static difficulty buckets and online learner state, and this can improve selected task distributions, but the current initialization and migration policy are not yet robust enough for a broad final-performance claim.

## 7. Limitations

The current project has several important limitations.

1. The strongest experiments appear to be limited-run comparisons rather than multi-seed evaluations with confidence intervals.
2. The current initialization is difficulty-based. Difficulty labels are not necessarily aligned with the model's actual competence.
3. Rebucketing changes task composition across clusters. This can create unintended task imbalance.
4. The accuracy-based no-rebucket result is available, but the accuracy-based rebucketing result is still pending.
5. The current report uses accuracy-style task pass rates; more detailed reward, format-validity, and per-difficulty analyses should be promoted into the main results.
6. The migration direction rule is conservative and interpretable, but it may not be optimal for every task or every meaning of high absolute advantage.
7. The notebook and report use checked-in summaries rather than rerunning expensive RL training.

## 8. Future Work

The highest-priority next step is completing the accuracy-initialized rebucketing run. Instead of grouping examples only by nominal difficulty, DURCL should initialize buckets using observed model accuracy or pass rate per task/difficulty bucket. This should reduce noisy drift and make rebucketing corrections easier to interpret causally.

Additional follow-ups:

1. Run multi-seed comparisons for random, no-rebucket scheduler, and re-bucket scheduler.
2. Add task-balanced rebucketing constraints or report task-balance metrics as first-class outputs.
3. Separate within-task difficulty correction from cross-task composition shifts.
4. Compare difficulty-init against accuracy-init using the same checkpoints, seeds, and evaluation sets.
5. Track per-difficulty performance, not only aggregate task accuracy.
6. Evaluate whether reverse migration should remain disabled or be introduced under stronger guardrails.
7. Report compute cost and convergence speed, since curriculum scheduling is partly a sample-efficiency method.

## 9. Conclusion

DURCL is a practical adaptive curriculum mechanism for RL-based LLM post-training over Math+Zebra environments. Its UCB sampler and micro-bucket rebucketing module are implemented and tested at the mechanism level. The present experiments show promising structural evidence and step-200 gains, but they do not yet justify a broad claim of overall superiority. The main scientific contribution of the current project is diagnosing how static difficulty buckets can mismatch the learner's online state and showing that rebucketing can produce interpretable corrections. The next stage should complete accuracy-initialized rebucketing and stricter evaluation to determine whether these structural corrections translate into robust final-performance gains.

## References

1. Zhenting Wang, Guofeng Cui, Kun Wan, and Wentian Zhao. "DUMP: Automated Distribution-Level Curriculum Learning for RL-based LLM Post-training." arXiv:2504.09710, 2025. https://arxiv.org/abs/2504.09710
2. Yoshua Bengio, Jerome Louradour, Ronan Collobert, and Jason Weston. "Curriculum Learning." ICML 2009. https://icml.cc/2009/papers/119.pdf
3. Peter Auer, Nicolo Cesa-Bianchi, and Paul Fischer. "Finite-time Analysis of the Multiarmed Bandit Problem." Machine Learning, 2002. https://www.cs.utexas.edu/~shivaram/readings/b2hd-AuerCF2002.html
4. John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, and Oleg Klimov. "Proximal Policy Optimization Algorithms." arXiv:1707.06347, 2017. https://arxiv.org/abs/1707.06347
5. Qwen Team. "Qwen2.5 Technical Report." arXiv:2412.15115, 2024. https://arxiv.org/abs/2412.15115
6. verl community. "verl: Volcano Engine Reinforcement Learning for LLMs." https://github.com/agentica-project/verl

## Repo Evidence Map

- Project status and high-level results: `README.md`
- Difficulty-init and re-bucket analysis: `docs/result_report.md`
- Scheduler design notes: `docs/scheduler_v1.md`
- Scheduler unit-test plan: `docs/scheduler_test_plan.md`
- Scheduler implementation: `src/scheduler/adaptive_curriculum_scheduler.py`
- Scheduler unit tests: `tests/test_scheduler_unit.py`
- Toy simulation summary: `artifacts/toy_rebucket_summary.json`
- Math+Zebra no-rebucket results: `experiments/results/math_zebra_2data/README.md`
