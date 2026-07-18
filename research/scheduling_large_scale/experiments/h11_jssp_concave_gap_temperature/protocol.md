# H11: Mean-One Concave Relative-Gap Temperature for JSSP50x20

## Status

Preregistered only after H10 completed cleanly below the fixed matched-BOPO
promotion gate. Launch exactly one smoke, then one screen only if the smoke
passes. Never reuse an existing output or log target.

## Observed Motivation

H8's linear mean-one relative-gap temperature reached validation32
`3080.624992` at step500, `4.687523` below matched BOPO `3085.312515`, but the
paired effect was heterogeneous: 15/32 instances favored H8 and the bootstrap
95% CI was `[-13.374925, 3.187508]`.

H10 removed the relative-gap temperature and substituted ASW gap-regret
weights. Its best validation32 mean was step250 at `3088.687592`, `3.375076`
above BOPO. Thus H8's gap emphasis is necessary for the matched-mean gain, but
its linear temperature is the remaining bounded source of extreme-pair
pressure that may amplify cross-instance heterogeneity.

## Hypothesis

Retain H8's source-normalized base logit and relative-gap ordering, but replace
the linear temperature with a mean-one square-root temperature:

`base = 100 * (log_prob_w_mean - log_prob_l_mean) / mean_abs_gap`

`raw_temperature = sqrt(abs(advantage_gap) / mean_abs_gap)`

`temperature = raw_temperature / mean(raw_temperature)`

`logit = base * temperature`

The temperature still has mean one and remains monotone in pair gap, but its
concave curvature compresses the widest comparisons. This should retain H8's
mean optimization while reducing outlier-pair pressure and improving response
consistency across validation instances.

This is a separately labelled concave-temperature USW variant, not the
original USW claim and not an H8 retry.

## Locked Protocol

- Common checkpoint: `downloads/jssp15x15/weighting/checkpoint.ckpt`.
- Seed and ordered dynamic training stream: `12345678`, start index 0.
- Validation stream: identical locked H3-H10 validation32 stream.
- JSSP50x20, FP32, physical instance batch one.
- B=128 rollouts and K=16 selected candidates per instance.
- Exactly 15 rank-stratified best-anchor pairs, strictly instance-local.
- H8 builder remains byte-for-byte unchanged.
- The only change from H8 is linear-to-square-root relative-gap temperature;
  source factor100, global mean-gap normalization, uniform weights, sigmoid
  link, normalized aggregation, optimizer, and all data streams remain fixed.
- Fresh Adam, LR `1e-5`, weight decay `1e-6`.
- Fixed matched BOPO threshold: `3085.312515`; lower is better.
- H8 step500 `3080.624992` remains the current family mean leader.
- No paired inference, TA, DMU, or final-test read is allowed for selection.

## Stages and Gates

1. Run exactly one fresh-root update. Require one step-0 and one step-1
   validation, finite loss, finite strictly positive gradient, B128/K16,
   physical batch one, exactly 15 instance-local pairs, and a clean exact-log
   error scan.
2. Only after smoke passes, run one fresh-root 300-update screen with
   validation at steps0/50/100/150/200/250/300.
3. Continue to the matched step500 budget only if a scheduled screen
   checkpoint is strictly below `3085.312515`. Otherwise H11 is negative and
   must not be retried or continued.
4. Run paired validation inference only for a validation-selected checkpoint
   below BOPO. A paper-facing win still requires a bootstrap CI wholly below
   zero and Holm-adjusted Wilcoxon `p<0.05`.

## Failure Interpretation

If H11 misses BOPO or remains paired-inconclusive, concave compression of H8's
gap temperature is insufficient. The next bounded family must target explicit
instance-level robustness; it must not repeat H4-H11 LR, weight decay, alpha,
pair geometry, source normalization, ASW weighting, or temperature-curvature
settings.
