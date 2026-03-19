from pathlib import Path

import torch

from rl4co.models.zoo.earl.mechanism import (
    MechanismConfig,
    MechanismProbe,
    MechanismTaskAdapter,
    cvrp_solution_to_edge_set,
    op_solution_to_node_set,
)


def test_cvrp_edge_set_keeps_depot_edges():
    solution = torch.tensor([1, 2, 0, 3], dtype=torch.long).numpy()
    edges = cvrp_solution_to_edge_set(solution)
    assert (0, 1) in edges
    assert (1, 2) in edges
    assert (0, 2) in edges
    assert (0, 3) in edges


def test_op_gain_uses_maximize_direction():
    adapter = MechanismTaskAdapter("op", objective_sense="max", diversity_metric="node_jaccard")
    score0 = torch.tensor([1.0, 2.0])
    scorek = torch.tensor([3.0, 2.5])
    gain = adapter.pair_gain(score0, scorek)
    assert torch.allclose(gain, torch.tensor([2.0, 0.5]))


def test_probe_writes_generic_mechanism_csv(tmp_path: Path):
    config = MechanismConfig(
        enabled=True,
        variant="eam",
        log_every=1,
        save_root=str(tmp_path),
        diversity_metric="edge_jaccard",
        objective_sense="min",
        seed=7,
        backbone="pomo",
    )
    probe = MechanismProbe(task_name="tsp", size=100, backbone="pomo", config=config)

    tau0 = torch.tensor([[0, 1, 2], [0, 2, 1]], dtype=torch.long)
    tauk = torch.tensor([[0, 2, 1], [0, 1, 2]], dtype=torch.long)
    score0 = torch.tensor([5.0, 6.0])
    scorek = torch.tensor([4.0, 5.0])
    ll0 = torch.tensor([-3.0, -4.0])
    llk = torch.tensor([-2.0, -3.0])

    stats = probe.compute(
        variant="eam",
        batch_size=2,
        tau0=tau0,
        tauk=tauk,
        pair_count=1,
        score0=score0,
        scorek=scorek,
        log_likelihood0=ll0,
        log_likelihoodk=llk,
        population_actions=None,
        step=9,
        epoch=1,
        val_metric=12.5,
    )
    csv_path = probe.dump(stats)

    assert csv_path == tmp_path / "tsp100_pomo" / "seed7" / "eam.csv"
    content = csv_path.read_text(encoding="utf-8")
    assert "task,size,backbone,variant,seed,step,epoch" in content
    assert "tsp,100,pomo,eam,7,9,1" in content


def test_op_node_jaccard_repr_ignores_depot():
    node_set = op_solution_to_node_set(torch.tensor([0, 4, 2, 0, 3], dtype=torch.long).numpy())
    assert node_set == {2, 3, 4}


def test_probe_respects_unpaired_gain_setting():
    config = MechanismConfig(
        enabled=True,
        variant="eam",
        log_every=1,
        save_root="outputs/mechanism",
        diversity_metric="edge_jaccard",
        objective_sense="min",
        paired_gain=False,
    )
    probe = MechanismProbe(task_name="tsp", size=20, backbone="pomo", config=config)
    stats = probe.compute(
        variant="eam",
        batch_size=1,
        tau0=torch.tensor([[0, 1, 2], [0, 2, 1]], dtype=torch.long),
        tauk=torch.tensor([[0, 2, 1], [0, 1, 2]], dtype=torch.long),
        pair_count=1,
        score0=torch.tensor([5.0, 7.0]),
        scorek=torch.tensor([4.0, 5.0]),
        log_likelihood0=torch.tensor([-3.0, -5.0]),
        log_likelihoodk=torch.tensor([-2.0, -4.0]),
        population_actions=None,
        step=0,
        epoch=0,
        val_metric=None,
    )
    assert stats["gain"] == 1.5
