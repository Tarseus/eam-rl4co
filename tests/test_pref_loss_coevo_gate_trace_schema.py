from __future__ import annotations

import json
from pathlib import Path


def _import_loop(monkeypatch):
    repo_root = Path(__file__).resolve().parents[1]
    monkeypatch.syspath_prepend(str(repo_root / "PTP"))
    import ptp_discovery.pref_loss_coevo_loop as loop

    return loop


def test_extract_builder_cost_reads_pair_count_from_checks(monkeypatch):
    loop = _import_loop(monkeypatch)
    rec = {
        "builder_gate_trace": {
            "checks": [
                {"metric_name": "coverage", "observed_value": 1.0},
                {"metric_name": "pair_count", "observed_value": 960},
            ]
        }
    }
    assert loop._extract_builder_cost(rec) == 960.0


def test_build_pair_descriptor_reads_checks_metrics(monkeypatch):
    loop = _import_loop(monkeypatch)
    desc = loop._build_pair_descriptor(
        builder_gate_trace={
            "checks": [
                {"metric_name": "coverage", "observed_value": 1.0},
                {"metric_name": "pair_count", "observed_value": 960},
                {"metric_name": "semantic_pass_rate", "observed_value": 1.0},
            ]
        },
        proxy_agg={"proxy_effective_grad_ratio_mean": 0.5, "proxy_loss_mean": 2.5},
        pair_count_cap=1024,
        loss_scale=5.0,
        bins=8,
    )
    assert desc["g"]["coverage"] == 1.0
    assert desc["g"]["pair_count"] == 960
    assert desc["g"]["semantic_pass_rate"] == 1.0


def test_truncate_jsonl_by_generation_keeps_only_past_generations(monkeypatch, tmp_path):
    loop = _import_loop(monkeypatch)
    path = tmp_path / "records.jsonl"
    lines = [
        json.dumps({"generation": 0, "id": "a"}),
        json.dumps({"generation": 1, "id": "b"}),
        json.dumps({"generation": 2, "id": "c"}),
        "not_json",
        json.dumps({"id": "missing_generation"}),
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    kept, dropped = loop._truncate_jsonl_by_generation(str(path), 2)

    assert kept == 2
    assert dropped == 3

    kept_rows = [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]
    assert [int(r["generation"]) for r in kept_rows] == [0, 1]

