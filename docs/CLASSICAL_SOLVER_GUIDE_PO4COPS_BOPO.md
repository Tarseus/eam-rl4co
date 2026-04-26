# Classical Solver Guide for PO4COPs and BOPO Problems

This note summarizes how to solve the main problem families in this repository with traditional heuristic solvers or exact solvers, using the problem setups emphasized by:

- PO4COPs: Preference Optimization for Combinatorial Optimization Problems
- BOPO: Neural Combinatorial Optimization via Best-anchored and Objective-guided Preference Optimization

The goal is practical: if you want a strong non-neural baseline, a near-oracle reference, or a small exact benchmark set, this file tells you what to use first.

This version only covers the four problem families that matter here:

- `TSP`
- `CVRP`
- `FFSP`
- `JSSP`

`FJSP` is intentionally excluded from this note.

## 1. What the two papers use

### PO4COPs

For routing and FFSP, PO4COPs explicitly compares against classical solvers:

- Routing: `LKH3`, `HGS`, `Concorde`
- FFSP: `CPLEX`

In the paper's main tables:

- For `TSP20/50/100`, `Concorde` and `LKH3` are the classical references.
- For `CVRP20/50/100`, `LKH3` and `HGS` are the classical references.
- For `FFSP20/50/100`, the paper reports `CPLEX` plus simple classical heuristics such as `Random`, `Shortest Job First`, `Genetic Algorithm`, and `Particle Swarm Optimization`.

### BOPO

BOPO covers TSP and JSP in the parts relevant to this repository, and explicitly compares against classical or exact methods:

- TSP: `Concorde`, `Gurobi`, `LKH3`
- JSP: `Gurobi`, `Google OR-Tools`, and classical priority dispatching rules `SPT`, `MOR`, `MWR`

## 2. Definitive solver pairs

To keep this actionable, each problem below is assigned at least two concrete solvers. The first is the solver I recommend as the default baseline in this repository; the second is the solver I recommend as the stronger cross-check.

| Problem | Solver 1 | Role of solver 1 | Solver 2 | Role of solver 2 |
|---|---|---|---|---|
| `TSP` | `LKH-3` | Default high-quality heuristic baseline for `tsp50/tsp100` | `Concorde` | Exact solver for smaller subsets or certificate runs |
| `CVRP` | `LKH-3` | Paper-aligned heuristic baseline for `cvrp50/cvrp100` | `HGS` | Second paper-aligned heuristic baseline for `cvrp50/cvrp100` |
| `FFSP` | `Shortest Job First` | Cheap traditional heuristic baseline reported in PO4COPs | `OR-Tools CP-SAT` | Open-source exact/anytime replacement for the paper's `CPLEX` runs |
| `JSSP` | `job_shop_lib` | Default classical heuristic baseline via dispatching rules such as `SPT`, `MOR`, `MWR` | `OR-Tools CP-SAT` | Strong exact/anytime reference for benchmark subsets |

If you want a third option for robustness:

- `TSP`: `SCIP`
- `CVRP`: `PyVRP` as a maintained HGS-family implementation
- `FFSP`: `PyJobShop`
- `JSSP`: `PyJobShop`

## 3. Detailed guidance

### 3.1 TSP

#### Use these two solvers

- `LKH-3`: default TSP solver in this project
- `Concorde`: exact cross-check solver in this project

#### Why

- Both PO4COPs and BOPO treat `LKH3` as the main strong classical TSP heuristic.
- Both papers use exact TSP references when they want oracle-quality comparisons.
- For your repository's `TSP50` and `TSP100`, `LKH-3` is the most natural classical baseline.

#### What I would do in this repo

- Use `LKH-3` as the default traditional solver for `tsp50` and `tsp100`.
- Use `Concorde` on smaller evaluation subsets or whenever you need certified optimality.
- Use `SCIP` only as a third fallback when Concorde is inconvenient to build or integrate.

#### Data interface

Your routing data currently lives under:

- `data/tsp`

Most classical solvers expect `TSPLIB`-style text files, so the usual workflow is:

1. Load one instance from your `.npz`.
2. Export coordinates to a `TSPLIB` `.tsp` file.
3. Run `LKH-3` or `Concorde`.

#### Deterministic choice

- Primary solver: `LKH-3`
- Secondary solver: `Concorde`
- Optional third solver: `SCIP`

### 3.2 CVRP

#### Use these two solvers

- `LKH-3`: first CVRP solver in this project because PO4COPs explicitly uses it on CVRP
- `HGS`: second CVRP solver in this project because PO4COPs explicitly uses it on CVRP

#### Why

- PO4COPs explicitly evaluates CVRP against `LKH3` and `HGS` in both the baseline description and the CVRPLib table.
- `LKH-3` and `HGS` are therefore the paper-faithful choices.
- `PyVRP` is useful in practice because it is a maintained open-source implementation in the HGS family.
- `OR-Tools Routing` is convenient, but it is not the paper-faithful main baseline here.

#### What I would do in this repo

- Use `LKH-3` as the first traditional baseline for `cvrp50` and `cvrp100`.
- Use `HGS` as the second traditional baseline for `cvrp50` and `cvrp100`.
- If you want a maintained implementation rather than rebuilding the original HGS code path, use `PyVRP` as the practical substitute for `HGS`.
- Do not use `VRPSolverEasy` as the main CVRP comparison in this document, because that is not how PO4COPs sets up its CVRP classical baselines.

#### Data interface

Your CVRP data currently lives under:

- `data/vrp`

Typical workflow:

1. Load one `.npz` instance.
2. Export to a `CVRPLIB`-style `.vrp` file.
3. Solve it with `LKH-3` or `HGS`.

#### Deterministic choice

- Primary solver: `LKH-3`
- Secondary solver: `HGS`
- Optional third solver: `PyVRP`

### 3.3 FFSP

#### Use these two solvers

- `Shortest Job First`: first traditional FFSP solver in this project because PO4COPs reports it directly
- `OR-Tools CP-SAT`: second FFSP solver in this project as the open-source replacement for the paper's `CPLEX`

#### Why

- PO4COPs evaluates FFSP against `CPLEX` and also reports simple traditional heuristics like `Shortest Job First`, `Genetic Algorithm`, and `Particle Swarm Optimization`.
- `CPLEX` is commercial, so the most realistic open-source substitute is a `CP-SAT` model.
- `PyJobShop` already has flow-shop and hybrid-flow-shop style modeling support, which is a much better long-term choice than reviving one-off GA/PSO code from old repos.

#### What I would do in this repo

- For `ffsp50` and `ffsp100`, keep `Shortest Job First` as the first paper-faithful traditional heuristic baseline.
- Use `OR-Tools CP-SAT` as the open-source exact/anytime cross-check replacing `CPLEX`.
- If you want a higher-level modeling layer on top of CP-SAT, use `PyJobShop` as a third practical tool.

#### Data interface

Your FFSP environment config is here:

- `configs/env/ffsp.yaml`

It uses:

- `num_stage`
- `num_machine`
- `num_job`

You will usually need to convert each instance into stage-wise processing-time arrays for a CP-SAT or PyJobShop model.

#### Deterministic choice

- Primary solver: `Shortest Job First`
- Secondary solver: `OR-Tools CP-SAT`
- Optional third solver: `PyJobShop`

### 3.4 JSSP

#### Use these two solvers

- `job_shop_lib`: default JSSP classical baseline package
- `OR-Tools CP-SAT`: exact/anytime cross-check solver in this project

#### Why

- BOPO explicitly uses `SPT`, `MOR`, and `MWR` as representative traditional priority dispatching rules for JSP.
- BOPO also compares against exact solvers `Gurobi` and `Google OR-Tools`.
- In open-source practice, `OR-Tools CP-SAT` is the most straightforward replacement for commercial exact solvers on small and medium JSSP.

#### What I would do in this repo

- For cheap classical baselines on `jssp10x10`, `jssp15x15`, and `jssp20x20`, run `SPT`, `MOR`, and `MWR` through `job_shop_lib`.
- For stronger references, use `OR-Tools CP-SAT` with a time limit.
- Use `PyJobShop` only as a third alternative if you want one scheduling framework shared with `FJSP` and `FFSP`.

#### Data interface

Your JSSP benchmark files are already in `.jsp`-style text form:

- `data/jssp_bopo`

You also already have an evaluation entry point for neural models:

- `scripts/eval_jssp_benchmarks.py`

That means JSSP is the easiest problem family here to connect to classical solvers.

#### Deterministic choice

- Primary solver: `job_shop_lib`
- Secondary solver: `OR-Tools CP-SAT`
- Optional third solver: `PyJobShop`

## 4. Which solver should you choose first?

If your goal is fair comparison against the neural models in this repository, use:

1. `TSP50/TSP100`: `LKH-3`, and also run `Concorde` on a smaller subset
2. `CVRP50/CVRP100`: `LKH-3`, and also run `HGS`
3. `FFSP50/FFSP100`: `Shortest Job First`, and also run `OR-Tools CP-SAT`
4. `JSSP10x10/15x15/20x20`: `job_shop_lib`, and also run `OR-Tools CP-SAT`

If your goal is exact certificates on a smaller subset, use:

1. `TSP`: `Concorde`
2. `CVRP`: there is no paper-used exact solver here; keep the comparison heuristic with `LKH-3` and `HGS`, or add a separate exact study outside the paper protocol
3. `JSSP`: `OR-Tools CP-SAT`
4. `FFSP`: `OR-Tools CP-SAT`

## 5. Practical caveats

- `Concorde` and `LKH-3` are the most paper-aligned for TSP, but their licensing and build workflow are not as convenient as pure-Python packages.
- `HGS` is paper-faithful for CVRP, but `PyVRP` is often the easier maintained implementation path in practice.
- For `CVRP100`, exact solving is much more expensive than `TSP100`; do not expect exact runs on all 10k neural-test instances to be practical.
- For `FFSP100` and `JSSP20x20`, exact optimality is often unrealistic at scale. Use time-limited anytime search and compare objective plus runtime.
- For scheduling, a short-time CP-SAT run is often a better open-source baseline than a weak legacy heuristic repo.

## 6. My concrete recommendation for this project

If I were wiring traditional baselines into this repository, I would prioritize in this order:

1. `LKH-3` plus `Concorde` for `TSP`
2. `LKH-3` plus `HGS` for `CVRP`
3. `Shortest Job First` plus `OR-Tools CP-SAT` for `FFSP`
4. `job_shop_lib` plus `OR-Tools CP-SAT` for `JSSP`

That gives you:

- one strong routing heuristic stack,
- one paper-faithful CVRP heuristic pair,
- one unified open-source scheduling stack,
- and one lightweight dispatching-rule baseline family aligned with BOPO.

## 7. Sources

Papers:

- PO4COPs OpenReview PDF: <https://openreview.net/pdf?id=Jwe5FJ8QGx>
- BOPO OpenReview PDF: <https://openreview.net/pdf/239d6c0a4379ddac4e524debb5e71111c3f6ca24.pdf>

Solver/documentation links:

- Concorde home: <https://math.uwaterloo.ca/tsp/concorde/>
- LKH-3 official page: <https://webhotel4.ruc.dk/~keld/research/LKH-3/>
- SCIP TSP example: <https://www.scipopt.org/doc-3.1.0/examples/TSP/>
- HGS-CVRP GitHub: <https://github.com/vidalt/HGS-CVRP>
- PyVRP GitHub: <https://github.com/PyVRP/PyVRP>
- PyVRP docs: <https://pyvrp.org/>
- OR-Tools routing overview: <https://developers.google.com/optimization/routing/routing_tasks>
- OR-Tools job shop guide: <https://developers.google.com/optimization/scheduling/job_shop>
- OR-Tools scheduling overview: <https://developers.google.com/optimization/scheduling>
- PyJobShop docs: <https://pyjobshop.org/stable/>
- PyJobShop flexible job shop example: <https://pyjobshop.org/stable/examples/flexible_job_shop.html>
- PyJobShop hybrid flow shop example: <https://pyjobshop.org/stable/examples/hybrid_flow_shop.html>
- PyJobShop permutation flow shop example: <https://pyjobshop.org/stable/examples/permutation_flow_shop.html>
- JobShopLib docs: <https://job-shop-lib.readthedocs.io/en/stable/>
- JobShopLib dispatching rules: <https://job-shop-lib.readthedocs.io/en/v1.2.0/api/job_shop_lib.dispatching.rules.html>
