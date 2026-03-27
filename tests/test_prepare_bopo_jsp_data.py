from pathlib import Path

from scripts.prepare_bopo_jsp_data import _shape_counts, prepare_bopo_jsp


def test_shape_counts_groups_by_shape_prefix(tmp_path: Path) -> None:
    for name in [
        "10x10_0.jsp",
        "10x10_1.jsp",
        "15x10_0.jsp",
        "20x20_0.jsp",
    ]:
        (tmp_path / name).write_text("", encoding="utf-8")

    counts = _shape_counts(tmp_path)

    assert counts == {"10x10": 2, "15x10": 1, "20x20": 1}


def test_prepare_bopo_jsp_syncs_into_repo_owned_target(tmp_path: Path) -> None:
    source_root = tmp_path / "BOPO" / "JSP"
    target_root = tmp_path / "data" / "jssp_bopo"
    for rel_path in [
        "dataset5k/10x10_0.jsp",
        "benchmarks/validation/10x10_100.jsp",
        "benchmarks/TA/ta01.jsp",
        "benchmarks/LA/la01.jsp",
        "benchmarks/DMU/dmu01.jsp",
    ]:
        file_path = source_root / rel_path
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text("2 2\n0 1 1 2\n1 3 0 4\n11\n", encoding="utf-8")

    summary = prepare_bopo_jsp(tmp_path, source_root=source_root, target_root=target_root)

    assert summary["root"]["target_root"] == target_root.as_posix()
    assert (target_root / "train" / "10x10_0.jsp").is_file()
    assert (target_root / "validation" / "10x10_100.jsp").is_file()
    assert (target_root / "TA" / "ta01.jsp").is_file()
    assert (target_root / "LA" / "la01.jsp").is_file()
    assert (target_root / "DMU" / "dmu01.jsp").is_file()
