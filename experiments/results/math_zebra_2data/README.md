# Math+Zebra 2-data Results

## Random Baseline
- step 100: `math_train=0.440`, `zebra_train=0.263`
- step 200: `math_train=0.460`, `zebra_train=0.250`

## Scheduler Without Rebucketing
- step 60: `math_train=0.440`, `zebra_train=0.250`
- step 100: `math_train=0.420`, `zebra_train=0.237`
- step 200: `math_train=0.480`, `zebra_train=0.338`

## Final Comparison at Step 200
- math: scheduler `0.480` vs baseline `0.460` (`+0.020`)
- zebra: scheduler `0.338` vs baseline `0.250` (`+0.088`)

## No-Rebucket Cluster A_mean Trend
Saved in:
- `norebucket_cluster_timeseries_step1.csv`
- `norebucket_Amean_snapshots.csv`

Representative A_mean snapshots (`cluster_0..4`):
- step 60: `[0.329, 0.301, 0.259, 0.238, 0.230]`
- step 80: `[0.250, 0.264, 0.243, 0.252, 0.235]`
- step 100: `[0.190, 0.210, 0.250, 0.256, 0.220]`
- step 120: `[0.135, 0.189, 0.204, 0.275, 0.216]`
- step 140: `[0.072, 0.148, 0.176, 0.275, 0.257]`
- step 160: `[0.101, 0.141, 0.184, 0.233, 0.255]`
- step 180: `[0.034, 0.106, 0.197, 0.225, 0.264]`
- step 200: `[0.081, 0.101, 0.182, 0.295, 0.253]`
