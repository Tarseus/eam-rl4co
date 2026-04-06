from __future__ import annotations

import importlib
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "PTP"):
    s = str(p)
    if s not in sys.path:
        sys.path.insert(0, s)


def test_hf_subprocess_env_maps_physical_cuda_to_logical_cuda0():
    loop = importlib.import_module("ptp_discovery.pref_loss_coevo_loop")

    env, worker_device = loop._hf_subprocess_env_and_device("cuda:3")

    assert env["CUDA_VISIBLE_DEVICES"] == "3"
    assert worker_device == "cuda:0"


def test_hf_subprocess_env_can_enable_cuda_launch_blocking():
    loop = importlib.import_module("ptp_discovery.pref_loss_coevo_loop")

    env, worker_device = loop._hf_subprocess_env_and_device(
        "cuda:1",
        {"hf_subprocess_cuda_launch_blocking": True},
    )

    assert env["CUDA_VISIBLE_DEVICES"] == "1"
    assert env["CUDA_LAUNCH_BLOCKING"] == "1"
    assert worker_device == "cuda:0"


def test_worker_device_fields_keep_physical_and_logical_devices():
    loop = importlib.import_module("ptp_discovery.pref_loss_coevo_loop")

    physical_device, logical_device = loop._worker_device_fields(
        {
            "device_physical_str": "cuda:2",
            "device_str": "cuda:0",
        }
    )

    assert physical_device == "cuda:2"
    assert logical_device == "cuda:0"


def test_hf_subprocess_failure_record_preserves_logical_device_metadata():
    loop = importlib.import_module("ptp_discovery.pref_loss_coevo_loop")

    rec = loop._hf_subprocess_failure_record(
        {
            "device_physical_str": "cuda:3",
            "device_str": "cuda:0",
        },
        reason="child_failed",
        error="boom",
    )

    assert rec["device"] == "cuda:3"
    assert rec["device_str"] == "cuda:3"
    assert rec["device_logical_str"] == "cuda:0"


def test_collect_cuda_snapshot_and_format_without_requested_devices():
    diag = importlib.import_module("ptp_discovery.cuda_diagnostics")

    snapshot = diag.collect_cuda_snapshot(devices=[], include_nvidia_smi=False)
    text = diag.format_cuda_snapshot(snapshot)

    assert isinstance(snapshot, dict)
    assert snapshot["requested_devices"] == []
    assert text.startswith("cuda_diag(")


def test_run_hf_pair_eval_sets_logical_device_after_cuda_visible_devices_remap(monkeypatch, tmp_path):
    worker = importlib.import_module("ptp_discovery.run_hf_pair_eval")

    set_device_calls: list[str] = []
    writes: list[dict] = []

    class DummyTrace:
        def __init__(self, *_args, **_kwargs):
            pass

        def start(self, **_kwargs):
            return None

        def install_signal_handlers(self):
            return None

        def heartbeat(self, **_kwargs):
            return None

        def finish(self, **_kwargs):
            return None

        def fail(self, **_kwargs):
            return None

        def close(self):
            return None

    monkeypatch.setattr(worker, "RuntimeTrace", DummyTrace)
    monkeypatch.setattr(
        worker,
        "_load_json",
        lambda _path: {
            "generation": 1,
            "pair_index": 2,
            "g_id": "g_ref",
            "f_id": "f_ref",
            "device_physical_str": "cuda:3",
            "device_str": "cuda:0",
        },
    )
    monkeypatch.setattr(worker, "_atomic_write_json", lambda _path, payload: writes.append(dict(payload)))
    monkeypatch.setattr(worker, "_evaluate_pair_worker", lambda payload: dict(payload, pair_ok=True, score=0.0))
    monkeypatch.setattr(worker.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(worker.torch.cuda, "set_device", lambda dev: set_device_calls.append(str(dev)))

    result_path = tmp_path / "result.json"
    exit_code = worker.main(["--payload", str(tmp_path / "payload.json"), "--result", str(result_path)])

    assert exit_code == 0
    assert set_device_calls == ["cuda:0"]
    assert writes[-1]["device_str"] == "cuda:0"
    assert writes[-1]["device_physical_str"] == "cuda:3"
