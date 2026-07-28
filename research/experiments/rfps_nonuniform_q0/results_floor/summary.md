# Non-uniform q0 results

- candidates: 40
- probes: 16
- all-pair/uniform max error: 1.481e-04

| Anchor | Condition | rho | false skip | NN median | E/F rank | E/F top-1 |
|---|---|---:|---:|---:|---:|---:|
| scratch | uniform | 0.336 | 0.111 | 0.00794 | 0.998 | 0.975 |
| scratch | anchor_is | 0.586 | 0.036 | 0.01072 | 0.996 | 0.925 |
| scratch | all_pair_endpoint | 0.390 | 0.111 | 0.00794 | 0.996 | 0.950 |
| scratch | gap_endpoint | 0.415 | 0.142 | 0.00821 | 0.994 | 0.975 |
| scratch | anchor_gap_endpoint | 0.590 | 0.036 | 0.01072 | 0.997 | 0.900 |
| scratch | anchor_hard_endpoint | 0.569 | 0.036 | 0.01072 | 0.997 | 0.925 |
| warm | uniform | 0.390 | 0.000 | 0.00167 | 1.000 | 0.950 |
| warm | anchor_is | 0.652 | 0.000 | 0.00164 | 0.995 | 0.875 |
| warm | all_pair_endpoint | 0.395 | 0.000 | 0.00155 | 1.000 | 0.975 |
| warm | gap_endpoint | 0.387 | 0.000 | 0.00135 | 1.000 | 0.950 |
| warm | anchor_gap_endpoint | 0.657 | 0.000 | 0.00170 | 0.996 | 0.875 |
| warm | anchor_hard_endpoint | 0.652 | 0.000 | 0.00170 | 0.994 | 0.875 |
