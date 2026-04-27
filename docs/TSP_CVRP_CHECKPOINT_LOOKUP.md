# TSP/CVRP Long-Run Checkpoint Lookup

Last updated: 2026-04-26

This note is for the TSP/CVRP runs that trained noticeably longer than the usual
`100`/`200` epochs and still need their final `test/max_reward` and
`test/max_aug_reward` checked or re-evaluated.

## Naming rule

From the saved `hparams.yaml`, the relevant long runs consistently use:

- checkpoint directory: `${paths.output_dir}/checkpoints`
- checkpoint filename pattern: `epoch_{epoch:03d}.ckpt`
- rolling latest checkpoint: `last.ckpt`
- best checkpoint selection: `save_top_k: 1`, monitored on `val/reward`

So if a run root is `logs/train/runs/<run_name>/...`, the first place to look is
usually:

```text
logs/train/runs/<run_root>/checkpoints/
```

In one important case, a mirrored full run tree also exists under `curves/`, and
that mirror already contains `.ckpt` files.

## Local Checkpoints Already Present

These runs already have local checkpoint files in the current workspace.

| Problem | Method | Curve alias | Canonical local run root | Local checkpoint dir | Files seen locally | Max epoch seen in metrics | Test metrics in CSV |
| --- | --- | --- | --- | --- | --- | ---: | --- |
| TSP100 | BOPO | `curves/tsp100_bopo.csv` | `curves/tsp100_cvrp100_baselines_20260407-030000/tsp100_bopo` | `curves/tsp100_cvrp100_baselines_20260407-030000/tsp100_bopo/checkpoints` | `epoch_214.ckpt`, `last.ckpt` | 358 | Missing |
| TSP100 | SLL | `curves/tsp100_sll.csv` | `curves/tsp100_cvrp100_baselines_20260407-030000/tsp100_sll` | `curves/tsp100_cvrp100_baselines_20260407-030000/tsp100_sll/checkpoints` | `epoch_294.ckpt`, `last.ckpt` | 298 | Missing |
| CVRP100 | BOPO | `curves/cvrp100_bopo.csv` | `curves/tsp100_cvrp100_baselines_20260407-030000/cvrp100_bopo` | `curves/tsp100_cvrp100_baselines_20260407-030000/cvrp100_bopo/checkpoints` | `epoch_070.ckpt`, `last.ckpt` | 226 | Missing |
| CVRP100 | SLL | `curves/cvrp100_sll.csv` | `curves/tsp100_cvrp100_baselines_20260407-030000/cvrp100_sll` | `curves/tsp100_cvrp100_baselines_20260407-030000/cvrp100_sll/checkpoints` | `epoch_229.ckpt`, `last.ckpt` | 234 | Missing |

### Local mirror note

There is also a duplicate BOPO mirror here:

- `curves/tsp100_cvrp100_baselines_20260407-030000/tsp100_cvrp100_bopo_20260410-064304/tsp100_bopo/checkpoints`
- `curves/tsp100_cvrp100_baselines_20260407-030000/tsp100_cvrp100_bopo_20260410-064304/cvrp100_bopo/checkpoints`

Those duplicated BOPO checkpoint folders also contain `epoch_214.ckpt` /
`epoch_070.ckpt` plus `last.ckpt`.

## Expected On Server, Missing Locally

These runs have long metrics traces, but the corresponding checkpoint directory is
not present in the current workspace. On the server, search the listed run root
for `checkpoints/last.ckpt` and `checkpoints/epoch_*.ckpt`.

| Problem | Method | Curve/log source | Expected run root on server | Expected checkpoint dir | Max epoch seen in metrics | Local status | Notes |
| --- | --- | --- | --- | --- | ---: | --- | --- |
| TSP50 | BOPO | `curves/tsp50_bopo.csv`; `logs/train/runs/tsp50_cvrp50_supplement_20260421-071651/tsp50_bopo/tsp50_bopo/version_0/metrics.csv` | `logs/train/runs/tsp50_cvrp50_supplement_20260421-071651/tsp50_bopo` | `logs/train/runs/tsp50_cvrp50_supplement_20260421-071651/tsp50_bopo/checkpoints` | 653 | Missing locally | `test/max_reward` and `test/max_aug_reward` both missing from CSV |
| CVRP50 | BOPO | `curves/cvrp50_bopo.csv`; `logs/train/runs/tsp50_cvrp50_supplement_20260421-071651/cvrp50_bopo/cvrp50_bopo/version_0/metrics.csv` | `logs/train/runs/tsp50_cvrp50_supplement_20260421-071651/cvrp50_bopo` | `logs/train/runs/tsp50_cvrp50_supplement_20260421-071651/cvrp50_bopo/checkpoints` | 342 | Missing locally | `test/max_reward` and `test/max_aug_reward` both missing from CSV |
| TSP50 | SLL | `curves/tsp50_sll.csv`; `logs/train/runs/tsp50_cvrp50_sll_20260413-025121/tsp50_sll_seed1234/tsp50_sll_seed1234/version_0/metrics.csv` | `logs/train/runs/tsp50_cvrp50_sll_20260413-025121/tsp50_sll_seed1234` | `logs/train/runs/tsp50_cvrp50_sll_20260413-025121/tsp50_sll_seed1234/checkpoints` | 427 | Missing locally | `test/max_reward` and `test/max_aug_reward` both missing from CSV |
| CVRP50 | SLL | `curves/cvrp50_sll.csv`; `logs/train/runs/tsp50_cvrp50_sll_20260413-025121/cvrp50_sll_seed1234/cvrp50_sll_seed1234/version_0/metrics.csv` | `logs/train/runs/tsp50_cvrp50_sll_20260413-025121/cvrp50_sll_seed1234` | `logs/train/runs/tsp50_cvrp50_sll_20260413-025121/cvrp50_sll_seed1234/checkpoints` | 322 | Missing locally | `test/max_reward` and `test/max_aug_reward` both missing from CSV |
| TSP100 | PO/base long run | `curves/tsp100_base.csv`; `curves/tsp100_po.csv`; `logs/train/runs/2026-02-16_09-12-57/csv/version_0/metrics.csv` | `logs/train/runs/2026-02-16_09-12-57` | `logs/train/runs/2026-02-16_09-12-57/checkpoints` | 710 | Missing locally | Curve has no terminal `test/*` rows despite long training |
| TSP100 | free_loss (`best` / `loss_only`) | `curves/tsp100_best.csv`; `curves/tsp100_loss_only.csv`; `logs/train/runs/my_loss_TSP100/csv/version_0/metrics.csv` | `logs/train/runs/my_loss_TSP100` | `logs/train/runs/my_loss_TSP100/checkpoints` | 204 | Missing locally | Curve aliases point to the same metrics trace |
| TSP100 | free_loss (`best_weighting`) | `curves/tsp100_best_weighting.csv`; `logs/train/runs/2026-04-16_14-57-36/csv/version_0/metrics.csv` | `logs/train/runs/2026-04-16_14-57-36` | `logs/train/runs/2026-04-16_14-57-36/checkpoints` | 258 | Missing locally | No `test/*` rows in the long-run CSV |
| TSP50 | free_loss (`best_weighting`) | `curves/tsp50_best_weighting.csv`; `logs/train/runs/2026-04-16_14-57-39/csv/version_0/metrics.csv` | `logs/train/runs/2026-04-16_14-57-39` | `logs/train/runs/2026-04-16_14-57-39/checkpoints` | 526 | Missing locally | No `test/*` rows in the long-run CSV |

## Curve Aliases Without A Recoverable Run Root

These curve files exist locally, but I could not map them back to a unique run
directory under `logs/train/runs/` from the current workspace alone. That means
their original checkpoint location must be recovered from server-side shell search,
older notebook history, or the original training launcher output.

| Curve file | Problem | Max epoch in curve | What is missing |
| --- | --- | ---: | --- |
| `curves/tsp50_best.csv` | TSP50 | 110 | No matching run root found locally, so checkpoint dir cannot be inferred reliably |
| `curves/tsp50_loss_only.csv` | TSP50 | 110 | Same trace as `tsp50_best.csv`; original checkpoint dir not recoverable from local artifacts |

## Practical Server Search Order

For the unresolved runs above, search in this order:

1. `logs/train/runs/<run_root>/checkpoints/last.ckpt`
2. `logs/train/runs/<run_root>/checkpoints/epoch_*.ckpt`
3. If the run is part of the `tsp100_cvrp100_baselines_20260407-030000` mirror, also check the mirrored `curves/.../checkpoints/`
4. If only a curve alias exists, grep or `find` on the server by the run timestamp or run name from the table above

## Fast Follow-up

The highest-value re-eval targets are:

1. `TSP50 BOPO` at `653` epochs
2. `TSP100 PO/base` at `710` epochs
3. `TSP50 free_loss best_weighting` at `526` epochs
4. `TSP50 SLL` at `427` epochs
5. `TSP100 BOPO/SLL` and `CVRP100 BOPO/SLL` from the local `curves/.../checkpoints/` mirrors
