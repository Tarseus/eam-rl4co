# H2 implementation note

`run_two_point_exact.py` is the authoritative H2 runner.

It represents every trajectory-step Fisher tangent coordinate and accumulates
candidate Gram matrices probe by probe. `run_two_point.py` contains the shared
closed-form coefficient and policy-step routines plus an earlier
trajectory-collapsed prototype. That prototype is retained for audit but its
`main()` is not a valid H2 evaluation because second-point step weights are
candidate-dependent and cannot be collapsed to one scalar per trajectory.

The committed `results/two_point.json` and `results/two_point.csv` were
generated only by `run_two_point_exact.py`.
