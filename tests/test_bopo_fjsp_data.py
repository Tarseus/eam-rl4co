from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import pytest


def _load_bopo_data_module():
    module_path = Path(__file__).resolve().parents[1] / "rl4co" / "models" / "zoo" / "bopo_fjsp" / "data.py"
    spec = spec_from_file_location("bopo_fjsp_data_only", module_path)
    module = module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_bopo_fjsp_padding_uses_invalid_ops(tmp_path):
    mod = _load_bopo_data_module()
    sample = "\n".join(
        [
            "2 2 0",
            "1 1 1 3",
            "2 1 1 2 1 2 4",
            "999",
        ]
    )
    fpath = tmp_path / "toy.fjs"
    fpath.write_text(sample, encoding="utf-8")

    _name, process_times, _makespan = mod._read_basic(str(fpath))

    # Job 0 has only one real operation, so the padded second operation must be
    # invalid on all machines instead of becoming a fake zero-cost machine-0 op.
    assert process_times[0, 1].tolist() == [-1, -1]


def test_bopo_fjsp_legacy_cache_is_rebuilt(tmp_path):
    mod = _load_bopo_data_module()
    sample = "\n".join(
        [
            "2 2 0",
            "1 1 1 3",
            "2 1 1 2 1 2 4",
            "999",
        ]
    )
    (tmp_path / "toy.fjs").write_text(sample, encoding="utf-8")

    # Simulate an old unversioned cache payload.
    legacy_instances = [{"legacy": True}]
    mod.torch.save(legacy_instances, tmp_path / "cached.pt")

    instances = mod.load_dataset(str(tmp_path), use_cached=True, device="cpu")
    assert isinstance(instances, list)
    assert instances and "legacy" not in instances[0]

    cached = mod.torch.load(tmp_path / "cached.pt", map_location="cpu", weights_only=False)
    assert cached["cache_version"] == mod._CACHE_VERSION


def test_bopo_fjsp_features_stay_finite_with_padded_jobs(tmp_path):
    mod = _load_bopo_data_module()
    sample = "\n".join(
        [
            "2 2 0",
            "1 1 1 3",
            "2 1 1 2 1 2 4",
            "999",
        ]
    )
    fpath = tmp_path / "toy.fjs"
    fpath.write_text(sample, encoding="utf-8")

    instance = mod.load_instance(str(fpath), device="cpu")
    assert mod.torch.isfinite(instance["x"]).all()


def test_bopo_fjsp_empty_edge_sets_keep_pyg_shape(tmp_path):
    mod = _load_bopo_data_module()
    sample = "\n".join(
        [
            "2 2 0",
            "1 1 1 3",
            "2 1 1 2 1 2 4",
            "999",
        ]
    )
    fpath = tmp_path / "toy.fjs"
    fpath.write_text(sample, encoding="utf-8")

    instance = mod.load_instance(str(fpath), device="cpu")
    assert tuple(instance["job_edges"].shape) == (2, 0)
    assert tuple(instance["ops_edges"].shape) == (2, 0)
    assert tuple(instance["mac_edges"].shape)[0] == 2


@pytest.mark.skipif(
    spec_from_file_location is None,
    reason="spec import machinery unavailable",
)
def test_bopo_fjsp_real_operation_count_matches_variable_lengths(tmp_path):
    module_path = Path(__file__).resolve().parents[1] / "rl4co" / "models" / "zoo" / "bopo_fjsp" / "sampling.py"
    spec = spec_from_file_location("bopo_fjsp_sampling_only", module_path)
    module = module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)

    sample = "\n".join(
        [
            "2 2 0",
            "1 1 1 3",
            "2 1 1 2 1 2 4",
            "999",
        ]
    )
    data_mod = _load_bopo_data_module()
    fpath = tmp_path / "toy.fjs"
    fpath.write_text(sample, encoding="utf-8")
    instance = data_mod.load_instance(str(fpath), device="cpu")

    assert module._count_real_operations(instance) == 3
