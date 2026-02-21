# EoH: LLM-Driven Evolutionary Free-Loss Discovery (RL4CO Backend)

This repository contains an **EoH** (Evolution-of-Heuristics style) discovery loop that uses an LLM to evolve **free-form preference losses** (`free_loss`) and evaluates them with the **RL4CO backend**.

The discovery loop produces:
- `best_candidate.json` (contains `ir` + fitness metrics)
- `checkpoint.json` (resume state)
- `gate_reports.jsonl` / `candidates.jsonl` / `fitness_scores.jsonl`

RL4CO training can directly consume `best_candidate.json` via `model.free_loss_ir_json_path`.

## Install (optional EoH deps)

EoH requires extra dependencies (kept out of the default install):

```bash
pip install -e ".[eoh]"
```

## Environment Variables

Set at least:
- `OPENAI_API_KEY` (required unless you explicitly run with `offline_mode=true`)

Optional:
- `OPENAI_BASE_URL` (default: `https://api.openai.com/v1`)
- `OPENAI_MODEL` (default in code: `gpt-4.1`)
- `OPENAI_TIMEOUT_S` (default: `60`)

Example:
```bash
export OPENAI_API_KEY="sk-..."
export OPENAI_MODEL="gpt-4.1"
```

## Run Discovery (RL4CO)

Use the provided script:

```bash
python scripts/run_free_loss_discovery_rl4co.py --config configs/experiment/free_loss_discovery/rl4co.yaml --device cpu
```

Outputs are written under:
- `runs/free_loss_discovery/<timestamp>/`

### Resume

```bash
python scripts/run_free_loss_discovery_rl4co.py --config configs/experiment/free_loss_discovery/rl4co.yaml --resume-latest
```

## LLM Cache

For robustness and cheaper re-runs, the LLM layer writes a run-scoped cache:
- `runs/free_loss_discovery/<run>/llm_cache.jsonl`

Re-running the same run (or resuming) should increase cache hits and reduce real requests.

## Use the Best Candidate in RL4CO Training

The discovery artifact `best_candidate.json` contains an `ir` object compatible with the free-loss compiler.

Point RL4CO training to it:

```bash
python run.py experiment=routing/pomo model.loss_type=free_loss model.free_loss_ir_json_path=runs/free_loss_discovery/<run>/best_candidate.json
```

The loader supports either:
- a plain IR JSON payload, or
- a wrapper JSON with an `ir` field (what `best_candidate.json` uses).

## Troubleshooting

### Prompts missing

Discovery config expects prompt files under `PTP/prompts/`. If they are missing, older versions would fall back to built-in prompts and log warnings.

Check:
- `PTP/prompts/free_loss_generation.txt`
- `PTP/prompts/free_loss_mutation.txt`
- `PTP/prompts/free_loss_crossover.txt`
- `PTP/prompts/free_loss_e2.txt`
- `PTP/prompts/free_loss_m2.txt`
- `PTP/prompts/free_loss_m3.txt`
- `PTP/prompts/free_loss_repair.txt`
- `PTP/prompts/free_loss_directed_repair.txt`
- `PTP/prompts/free_loss_expects_repair.txt`

### Dependency missing (`openai`, `python-dotenv`, `pyyaml`)

Install:
```bash
pip install -e ".[eoh]"
```

### OPENAI_API_KEY not set

The discovery loop will error with a clear message unless:
- you set `offline_mode=true` in the discovery config (which disables LLM calls).

### Not seeing E1/M1 triggered

E1/M1 correspond to LLM-driven crossover/mutation operations.

Check:
- `configs/experiment/free_loss_discovery/rl4co.yaml` includes prompt paths for `mutation` / `crossover`
- `gate_reports.jsonl` includes `llm_op` values like `M1`, `E1`, `E2`, etc.

### Gates fail frequently

Inspect:
- `runs/.../gate_reports.jsonl` for `static_error_code` / `dynamic_error_code`
- per-candidate logs: `runs/.../genXXX_candYYY.log`

Common causes:
- invalid `implementation_hint.expects`
- loss not finite / unstable gradients
- preference semantic violations

