from pathlib import Path

from rl4co.models.zoo.mgl_jssp.data import load_instance


def _write_bopo_like_jsp(path: Path) -> None:
    path.write_text(
        "\n".join(
            [
                "2 2",
                "0 3 1 5",
                "1 4 0 6",
                "11",
                "99 88",
                "",
            ]
        ),
        encoding="utf-8",
    )


def test_mgl_jssp_loader_supports_bopo_style_files(tmp_path: Path) -> None:
    file_path = tmp_path / "toy.jsp"
    _write_bopo_like_jsp(file_path)

    instance = load_instance(file_path.as_posix())

    assert instance["name"] == "toy"
    assert instance["shape"] == "2x2"
    assert float(instance["makespan"]) == 11.0
    assert tuple(instance["x"].shape) == (4, 15)
    assert tuple(instance["job_edges"].shape) == (2, 4)
    assert tuple(instance["mac_edges"].shape) == (2, 4)
    assert int(instance["machines"][0, 0].item()) == 0
    assert int(instance["machines"][0, 1].item()) == 1
