# TSP1000 USW/ASW Findings

## Research Question

How can USW or ASW be trained to significantly outperform PO on the same canonical TSP1000 fixed test?

## Current Understanding

USW and ASW are not uniformly inferior to PO: they learn faster at the beginning, but their pairwise logistic signal weakens as on-policy pairs become correctly ordered. The original transfer also changed problem size from TSP100 to TSP1000 and candidate-pool size from K=200 during discovery to K=20 during training. This combination makes saturation and weak preference estimation more plausible than an architectural expressivity limit.

The current inner loop therefore tests three mechanisms separately: larger instance-local candidate pools, lower logistic temperature scale, and detached ASW weights. If none closes the gap, the next intervention is a scheduled non-saturating PO component or hard-pair mining rather than an unstructured hyperparameter sweep.

## Key Results

- Canonical fixed100 baselines after 1000 continuation steps: PO 27.138046, BOPO 27.363355, USW 27.381906, ASW 27.561484. These are not valid baselines for the current 300-step screening sweep.
- PO beats final USW on 97/100 paired instances and final ASW on 100/100.
- USW validation is better than PO at continuation steps 100, 300, and 400, then PO pulls away.
- In the current K=40 exploratory sweep at continuation step 100, alpha=0.01 improves validation32 over alpha=0.05 for both USW (27.843909 vs 28.071091; -0.227183) and ASW (27.888632 vs 28.009202; -0.120570). This is screening evidence only, not canonical success.
- USW K64/alpha=0.01 is the current step-100 screening leader at 27.689156, improving another 0.154753 over USW K40/alpha=0.01. ASW detach is nearly neutral at step 100 (27.885151 detached vs 27.888632 differentiable).
- At continuation step 200, USW K40/alpha=0.01 reaches validation32 mean 27.300870. ASW K40/alpha=0.01 reaches 27.365960 without detach and 27.340087 with detach, while ASW K40/alpha=0.05 is 27.728695. USW K40/alpha=0.05 and USW K64/alpha=0.01 had not emitted their step-200 validations at this observation.
- The complete step-200 screen now puts USW K64/alpha=0.01 first at 27.149821, versus 27.300870 for K40/alpha=0.01 and 27.672419 for K40/alpha=0.05. The matched PO-300 control is 27.887167 at the same step. This is strong screening evidence for lower alpha and larger K, but it is not a final result.
- At the locked step-300 boundary, USW K64/alpha=0.01 wins the USW screen at validation32 mean 26.870726, ahead of K40/alpha=0.01 at 27.015497 and K40/alpha=0.05 at 27.560441. The matched PO-300 control is 27.538396. K64/alpha=0.01 is therefore the preregistered USW promotion.
- ASW K40/alpha=0.01 detached wins the ASW screen at step 300 with 27.030719, only 0.005744 better than non-detached 27.036463; alpha=0.05 reaches 27.611878. The tiny detach difference determines promotion under the locked rule but does not provide strong mechanistic evidence for H3.
- Both promoted `last.ckpt` files were verified at optimizer/global/epoch step 300 with nonempty Adam state (96 entries, all step 300), then resumed for 700 additional steps with `data_start_index=20038400`. Initial checks reached absolute steps 305 (USW) and 308 (ASW), both with replay error exactly zero.
- Encoder-only fused SDPA plus larger K raises trajectory throughput from 220.87 to as high as 427.42 trajectories/s.
- Decoder SDPA with a dynamic action mask is unsupported on the RTX 3090/CUDA 11.8 stack; decoder checkpointing remains mandatory.
- Fused encoder base batches above 128 fail strict rollout/replay equality and are excluded.

## Patterns and Insights

- PO's linear preference gradient remains approximately constant, while USW/ASW logistic coefficients and gradient norms shrink over training.
- More candidate paths are especially relevant for USW/ASW because the preference objective is estimated within each instance.
- ASW weights depend on absolute policy margin and are differentiable; this may emphasize already-easy pairs and permit loss reduction through weight changes.
- The first controlled K=40 comparison favors lower alpha in both objectives, consistent with the saturation hypothesis, but step-200/300 stability and fixed100 transfer remain unknown.
- Larger K currently adds proxy quality as well as trajectory throughput for USW, but it processes fewer unique instances per second; step 200/300 will determine whether the early gain persists.
- At step 200, increasing USW K from 40 to 64 at alpha=0.01 improves validation32 by 0.151049 despite reducing unique-instance throughput from about 8.45 to 6.27 instances/s. The K64 run still delivers about 401 trajectories/s, so the next decision depends on whether this advantage survives step 300.
- The USW K64 advantage survives step 300 and widens slightly to 0.144772 over K40/alpha=0.01, supporting the candidate-pool hypothesis despite lower unique-instance throughput.
- The ASW detach advantage shrinks from 0.025872 at step 200 to 0.005744 at step 300. Detach is promoted only because the preregistered metric ranks it first; the current evidence treats the mechanism as nearly neutral.

## Lessons and Constraints

- Do not declare success from the 8- or 32-instance validation mean; only canonical fixed100 paired evaluation counts.
- Do not compare a 300-continuation-step candidate to PO fixed100=27.138046. Use the matched PO-300 control for screening, and only apply the final PO threshold after the candidate has received 1000 continuation steps.
- Do not disable forced replay verification to accept faster kernels.
- Keep base instance batch at or below 128 with the current fused encoder.
- Limit g48 to four simultaneous TSP1000 training jobs until higher-concurrency power and thermal stability is demonstrated; a six-job launch rebooted the host.
- Compare configurations by wall time, trajectories processed, and unique instances processed—not optimizer steps alone.
- Existing six-run sweep is exploratory because it began before this workspace was initialized.
- When resuming a 300-step screen to total optimizer step 1000, advance `data_start_index` by `300 * batch_size`; the current script restores optimizer state but otherwise restarts the run-local data index at zero.
- The final equal-budget claim must evaluate `last.ckpt` at absolute optimizer step 1000, not an earlier validation-best checkpoint.
- Never reuse a completed run's output directory: a duplicate g48 process appended fresh step-0/100 records to `usw_k40_a005` after its valid step-300 summary. The duplicate was stopped, the valid step-300 result remains recoverable, and all promoted runs use new isolated directories.

## Open Questions

- Does K=40 or K=64 improve final fixed100 quality enough to offset lower unique-instance throughput?
- Is alpha=0.01 sufficient, or should temperature be normalized by sequence length or instance log-probability scale?
- Does detaching ASW weights help, or is the differentiable redistribution essential?
- If pure USW/ASW still plateaus, what PO mixing schedule preserves method identity while adding a non-saturating tail signal?

## Optimization Trajectory

Latest screening update: at continuation step 300, USW K64/alpha=0.01 leads validation32 at 26.870726 and detached ASW K40/alpha=0.01 leads its method at 27.030719, while matched PO-300 is 27.538396. The two winners are now running from absolute optimizer step 300 to 1000. These proxy results cannot satisfy the final claim; canonical fixed100 paired evaluation remains mandatory.

Current 1000-continuation-step canonical means: PO 27.138046 → USW 27.381906 / ASW 27.561484. The prior values are baselines for the original K20/alpha=0.05 configurations; the promoted K64 USW and detached K40 ASW candidates have not yet reached the final evaluation boundary.
