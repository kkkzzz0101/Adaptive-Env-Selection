# Result Report: Adaptive Env Selection (Difficulty-init Rebucket)

## 1) One-line takeaway

We test whether online rebucketing can correct imperfect initial curriculum buckets. Even with coarse difficulty-only initialization, rebucketing already learns meaningful redistribution patterns; acc-based initialization is the next step to obtain cleaner and potentially more beneficial corrections.

## 2) Key quantitative results

### 2.1 Scheduler trajectory snapshot (cluster-level weights)

- step 60: `[0.329, 0.301, 0.259, 0.238, 0.230]`
- step 80: `[0.250, 0.264, 0.243, 0.252, 0.235]`
- step 100: `[0.190, 0.210, 0.250, 0.256, 0.220]`
- step 120: `[0.135, 0.189, 0.204, 0.275, 0.216]`
- step 140: `[0.072, 0.148, 0.176, 0.275, 0.257]`
- step 160: `[0.101, 0.141, 0.184, 0.233, 0.255]`
- step 180: `[0.034, 0.106, 0.197, 0.225, 0.264]`
- step 200: `[0.081, 0.101, 0.182, 0.295, 0.253]`

### 2.2 Step-200 validate comparison

- random baseline (step 200):
  - `math_train = 0.460`
  - `zebra_train = 0.250`
- no-rebucket scheduler (step 200):
  - `math_train = 0.480`
  - `zebra_train = 0.338`

Interpretation: at step 200, scheduler without rebucket already outperforms random baseline on both tasks.

### 2.3 Rebucket 100 -> 300 run

- step 300 validate:
  - `math_train = 0.520`
  - `zebra_train = 0.287`

### 2.4 Baseline 200 -> 300 reference

- baseline final (200 -> 300):
  - `math_train = 0.560`
  - `zebra_train = 0.275`

### 2.5 Rebucket step-200 reference

- rebucket step 200 validate:
  - `math_train = 0.540`
  - `zebra_train = 0.287`

## 3) Cluster dynamics at rebucket step 300

- `cluster_0: A_mean=0.152, size=248, mig_in=6, mig_out=2`
- `cluster_1: A_mean=0.137, size=152, mig_in=7, mig_out=9`
- `cluster_2: A_mean=0.160, size=196, mig_in=8, mig_out=7`
- `cluster_3: A_mean=0.211, size=140, mig_in=4, mig_out=6`
- `cluster_4: A_mean=0.178, size=64,  mig_in=1, mig_out=2`

## 4) Composition drift analysis

### 4.1 Initial composition

By micro-bucket count:

| cluster | Math | Zebra | total |
|---|---:|---:|---:|
| C0 | 5 | 5 | 10 |
| C1 | 5 | 5 | 10 |
| C2 | 5 | 5 | 10 |
| C3 | 5 | 5 | 10 |
| C4 | 5 | 0 | 5 |

By sample count:

| cluster | Math | Zebra | total |
|---|---:|---:|---:|
| C0 | 80 | 100 | 180 |
| C1 | 80 | 100 | 180 |
| C2 | 80 | 100 | 180 |
| C3 | 80 | 100 | 180 |
| C4 | 80 | 0 | 80 |

Note: C4 is Math-only at initialization (Zebra has 4 difficulties per cluster, while Math has 5).

### 4.2 Final composition

By micro-bucket count:

| cluster | Math | Zebra | total |
|---|---:|---:|---:|
| C0 | 8 | 6 | 14 |
| C1 | 2 | 6 | 8 |
| C2 | 6 | 5 | 11 |
| C3 | 5 | 3 | 8 |
| C4 | 4 | 0 | 4 |

By sample count:

| cluster | Math | Zebra | total |
|---|---:|---:|---:|
| C0 | 128 | 120 | 248 |
| C1 | 32 | 120 | 152 |
| C2 | 96 | 100 | 196 |
| C3 | 80 | 60 | 140 |
| C4 | 64 | 0 | 64 |

### 4.3 Main shifts

- `C0: Math 80 -> 128, Zebra 100 -> 120`
- `C1: Math 80 -> 32,  Zebra 100 -> 120`
- `C2: Math 80 -> 96,  Zebra 100 -> 100`
- `C3: Math 80 -> 80,  Zebra 100 -> 60`
- `C4: Math 80 -> 64,  Zebra 0 -> 0`

Interpretation:

- C0 growth is largely driven by Math inflow.
- C1 becomes Zebra-heavy (strong task-composition skew).
- C3 loses Zebra significantly.
- C4 remains Math-only and shrinks.

This indicates composition drift: rebucketing is not only rebalancing within-task difficulty, but also altering task composition across clusters.

## 5) Main analysis direction for project report

### Paragraph 1: effectiveness under coarse initialization

Even with rough initialization based only on dataset difficulty levels, rebucketing still learns meaningful structural corrections. The migration is not purely random; it shows interpretable redistribution patterns (e.g., expansion of easier Math regions), suggesting partial recovery of learner-aligned grouping.

### Paragraph 2: why acc-based init is the next key step

If initialization is closer to learner state (via accuracy-based grouping), rebucketing corrections should be cleaner and less noisy, with reduced composition drift and stronger causal linkage to downstream gains. This is the core next-stage hypothesis for validating algorithm effectiveness.

## 6) Current status and next experiment

- Done: difficulty-init runs and composition-drift diagnostics (this report).
- In progress: acc-based rebucket runs.
- Next update: push acc-init comparison tables/plots once runs finish.
