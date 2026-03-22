# Preference Training Notes (RL4CO POMO)

This note documents tensor semantics and loss definitions used by RL4CO POMO.
Historical references to PTP reflect the original formulation, not an active backend.

## Tensor semantics

| Concept | RL4CO POMO |
| --- | --- |
| Reward | `reward: [B, P]` (usually `-cost`) |
| Trajectory log-prob | `log_likelihood: [B, P]` already equals sum log-prob |
| POMO dimension | `P` (num starts) |

Conclusion: RL4CO uses `log_likelihood` directly and does **not** need to re-sum per-step probabilities.

## Reward direction

RL4CO routing environments typically use `reward = -cost`, so higher reward means better solutions. The preference comparisons use `reward_i > reward_j` to mark winners (matching the historical PTP formulation).

## Loss types and inputs

### `rl_loss` (REINFORCE baseline)
- Inputs: `reward [B,P]`, `log_likelihood [B,P]`, shared baseline.
- Behavior: unchanged from RL4CO REINFORCE.

### `po_loss` (pairwise BT)
- Inputs: `reward [B,P]`, `log_likelihood [B,P]`.
- Preference matrix: `preference[i,j] = 1 if reward_i > reward_j else 0`.
- Logit difference: `logp_pair = alpha * (logp_i - logp_j)`.
- Loss: `-mean(logsigmoid(logp_pair) * preference)` over full `[B,P,P]`.
- Diagnostics: `po_pref_rate = mean(preference)`.

### `pl_loss` (listwise / ranking)
- Inputs: `reward [B,P]`, `log_likelihood [B,P]`.
- Sort by reward descending to get ranking.
- Apply `alpha` to `log_likelihood`.
- Loss (PTP-formulation): `mean(log(exp(logp)) - log(sum_exp))`, where `sum_exp`
  is a cumulative sum over the sorted list.
- Implementations:
- `pl_impl=ptp` uses the original one-hot + lower-triangular formulation.
  - `pl_impl=stable` uses an equivalent cumulative sum without `[B,P,P]` memory.

### `free_loss` (IR-defined, optional)
- Inputs: `reward [B,P]`, `log_likelihood [B,P]`, plus IR JSON file.
- Winner/loser pairs are constructed using `objective = -reward` (lower cost wins).
- When no pairs exist, fallback to a REINFORCE-style loss with mean baseline.

## Injection points in RL4CO

- Entry: `rl4co/models/zoo/pomo/model.py` (override `calculate_loss`).
- Loss helpers: `rl4co/models/rl/reinforce/preference_losses.py`.
- Optional free-form loss compiler: `rl4co/models/rl/reinforce/free_loss/`.
- Config: `configs/model/pomo.yaml` (loss_type, alpha, pl_impl, free_loss_ir_json_path, pref_builder_ir_json_path, pref_pair_json_path).

## Scheduling compatibility

- `rl4co.models.MatNet` inherits `POMO`, so FFSP MatNet and FJSP `*-pomo` scheduling experiments can switch to `loss_type=free_loss` without changing the trainer loop.
- `rl4co.models.StepwisePPO` does not expose `loss_type`, `free_loss_ir_json_path`, or preference-pair construction; current PPO scheduling experiments need algorithm changes before they can train with `free_loss`.
- FFSP is the cleanest full-pair setting in this repository because `FFSPEnv.get_num_starts()` returns `4! = 24`, giving `24 * 23 / 2 = 276` within-instance pairs for `all_pairs`.
- FJSP defaults to `get_num_starts() = 100`, so dense all-pairs grows to `100 * 99 / 2 = 4950` pairs per instance. For preference training, use a smaller explicit `num_starts` such as 16 or 24 unless you also cap builder density.
