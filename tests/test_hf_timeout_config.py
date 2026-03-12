import importlib.util
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "PTP"):
    s = str(p)
    if s not in sys.path:
        sys.path.insert(0, s)

spec = importlib.util.spec_from_file_location(
    "pref_loss_coevo_loop_test_module",
    ROOT / "PTP" / "ptp_discovery" / "pref_loss_coevo_loop.py",
)
assert spec is not None and spec.loader is not None
loop = importlib.util.module_from_spec(spec)
spec.loader.exec_module(loop)


def test_resolve_hf_timeout_disabled_by_default():
    assert loop._resolve_hf_timeout_s({}, default=None) is None
    assert loop._resolve_hf_timeout_s(None, default=None) is None


def test_resolve_hf_timeout_respects_positive_values():
    assert loop._resolve_hf_timeout_s({"high_fidelity_task_timeout_s": 3600}, default=None) == 3600.0
    assert loop._resolve_hf_timeout_s({"high_fidelity_task_timeout_s": "90"}, default=None) == 90.0


def test_resolve_hf_timeout_treats_non_positive_as_disabled():
    assert loop._resolve_hf_timeout_s({"high_fidelity_task_timeout_s": 0}, default=None) is None
    assert loop._resolve_hf_timeout_s({"high_fidelity_task_timeout_s": -1}, default=None) is None
