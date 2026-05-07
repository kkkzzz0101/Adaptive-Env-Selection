# Window Rebucket GlobalAcc (80 -> 200)

This directory collects the recoverable artifacts for the window-linear /
micro-bucket rebucketing run that resumed from the `warm80` checkpoint on the
remote machine `js2.blockelite.cn:10240`.

## Final metric summary

The consolidated metric table used by the notebook is:

- [`../final_rebucket_window_linear_80_200.csv`](../final_rebucket_window_linear_80_200.csv)

Its values are:

- step 100: `math=0.440`, `zebra=0.237`
- step 150: `math=0.520`, `zebra=0.237`
- step 200: `math=0.580`, `zebra=0.300`

Provenance:

- step 100 is consistent with a complete `global_step_100` micro-bucket
  checkpoint on the remote machine.
- step 150 and step 200 were manually recorded during the original run; no
  `global_step_150` directory was found on the recovered machine.
- the recovered `global_step_200` micro-bucket directory is incomplete: actor
  weights exist, but `curriculum_state.pt` and `data.pt` are missing, and the
  run's `latest_checkpointed_iteration.txt` still points to `100`.

## Checkpoint inventory

See [`checkpoint_inventory.csv`](checkpoint_inventory.csv).

Key observations:

- `scheduler_norebucket_mathzebra800_globalacc_warm80/global_step_80` is
  complete and serves as the source checkpoint.
- `scheduler_norebucket_mathzebra800_globalacc_from80_to200/global_step_200` is
  complete.
- `scheduler_microbucket_mathzebra800_globalacc_from80_to200/global_step_100`
  is complete.
- `scheduler_microbucket_mathzebra800_globalacc_from80_to200/global_step_200`
  contains actor weights but is not a complete recoverable scheduler-state
  checkpoint.

## Step 100 signature evidence

See [`step100_signature_summary.csv`](step100_signature_summary.csv).

This file was exported from the recovered
`scheduler_microbucket_mathzebra800_globalacc_from80_to200/global_step_100/curriculum_state.pt`
and records the window-signature statistics that drive rebucketing:

- `level`
- `slope`
- `volatility`
- `window_obs_count`
- candidate migration fields

Recovered summary statistics for the 40 micro-buckets at step 100:

- `window_obs_count`: min `196`, max `420`, avg `309.0`
- `slope`: min `-0.005767`, max `0.003274`, avg `-0.001090`
- `volatility`: min `0.117271`, max `0.333789`, avg `0.288033`
- candidate targets were set for `8` micro-buckets
- migration events completed so far: `0`
