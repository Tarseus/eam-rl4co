# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

RL4CO is an Extensive Reinforcement Learning (RL) for Combinatorial Optimization (CO) Benchmark that provides a unified framework for RL-based CO algorithms, decoupling science from engineering for reproducible research.

**Core Technologies:**
- PyTorch 2.6.0
- TorchRL 0.6.x-0.7.x
- TensorDict 0.6.x-0.7.x
- PyTorch Lightning 2.1+
- Hydra for configuration
- Hatchling build backend
- uv (recommended package manager)

## Common Commands

### Installation
- From PyPI: `pip install rl4co`
- From source: `pip install -U git+https://github.com/ai4co/rl4co.git`
- Local development (recommended):
  ```bash
  git clone https://github.com/ai4co/rl4co && cd rl4co
  uv sync --all-extras
  source .venv/bin/activate
  pre-commit install
  ```

### Training
- Default training (AM on TSP): `python run.py`
- Custom experiment: `python run.py experiment=routing/am env=cvrp env.num_loc=50`
- Hyperparameter sweep: `python run.py -m experiment=routing/am model.optimizer.lr=1e-3,1e-4,1e-5`
- Disable logging: `python run.py experiment=routing/am logger=none '~callbacks.learning_rate_monitor'`

### Testing
- Run all tests: `pytest tests`

### Code Style
- Pre-commit hooks run automatically on commit (black, ruff, etc.)
- Manually run: `pre-commit run --all-files`

## High-Level Architecture

```
rl4co/
├── envs/          # Environments for CO problems (routing, scheduling, eda, graph)
├── models/        # Policy and RL algorithm implementations
│   ├── zoo/       # Pre-built model zoo (constructive AR/NAR, improvement, transductive)
│   ├── rl/        # RL algorithms (REINFORCE, PPO, A2C)
│   └── networks/  # Neural network components
├── tasks/         # Training and evaluation tasks
└── utils/         # Utility functions

configs/           # Hydra configuration files
examples/          # Tutorial notebooks
tests/             # Test suite
```

**Key Architectural Patterns:**
- Modular design: Environments, policies, and RL algorithms are decoupled
- Policy types: Constructive (Autoregressive/NonAutoregressive), Improvement, Transductive
- Configuration via Hydra for composable, hierarchical configs
- Vectorized environments leveraging TorchRL for GPU acceleration

## Development Guidelines

- Follow the code style enforced by black and ruff via pre-commit hooks
- Write tests for new functionality in the `tests/` directory
- Use Hydra configs for experiment configuration rather than hardcoding
- Reference existing examples in the `examples/` directory for common patterns

## License

MIT License
