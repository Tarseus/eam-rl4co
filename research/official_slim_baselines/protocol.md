# Official SLIM Baseline Recovery Protocol

Status: locked before any official-SLIM smoke or training result is observed.

## Motivation

The repository's historical `sll_loss` is a custom listwise-ranking objective and
is not Corsini et al.'s Self-Labeling Improvement Method (SLIM).  All historical
artifacts labelled SLL/SLIM are therefore excluded as published-method baselines.

## Method identity

For each physical instance independently:

1. sample the full configured rollout pool;
2. select the first rollout attaining the best objective value;
3. use that trajectory only as a pseudo-label;
4. minimize its mean per-action cross-entropy;
5. average the resulting scalar losses over physical instances.

No ranking, pairwise margin, reward-gap weighting, or cross-instance selection is
allowed.  Rollout dimension and dataloader instance dimension remain distinct.

## Scope and stages

Primary paper recovery covers TSP50, TSP100, CVRP50, CVRP100, FFSP50, FFSP100,
JSSP10x10, and JSSP15x15.  TSP1000, CVRP1000, FFSP1000, and JSSP50x20 follow as
a separately labelled large-scale extension using each scale's already locked
matched continuation protocol.

For every backbone family, run one bounded smoke before full training.  A smoke
passes only with finite loss, finite nonzero gradient, correct pool size, strictly
instance-local pseudo-label selection, and a clean error scan.  Full runs use
fresh collision-free roots and do not overwrite historical `sll` artifacts.

## Fairness

- Reuse the corresponding historical baseline's architecture, initialization,
  data distribution, optimizer, learning-rate schedule, rollout count, physical
  instance batch, update/epoch budget, validation stream, and inference budget.
- Select the official-SLIM checkpoint on validation only.
- Evaluate once on the same fixed per-instance test streams used by the paper.
- Statistical comparisons are added to a newly preregistered multiplicity family;
  historical Holm-adjusted values are not silently reused.
- Do not call the cross-family CVRP/FFSP extension “published SLIM”; label it
  “SLIM objective on the matched backbone”.

## Compute allocation

At lock time g51 GPUs 5 and 7 are idle. GPUs 0--4 and 6 contain unrelated root
processes and are out of scope.  Existing processes and output roots are
authoritative and must not be killed, modified, or reused.
