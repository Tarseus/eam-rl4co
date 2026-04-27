from pathlib import Path

from rl4co.envs.scheduling.jssp.generator import JSSPFileGenerator
from rl4co.envs.scheduling.jssp.parser import read


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


def test_jssp_parser_supports_bopo_style_files(tmp_path: Path) -> None:
    file_path = tmp_path / "toy.jsp"
    _write_bopo_like_jsp(file_path)

    td, num_jobs, num_machines, max_ops_per_job = read(file_path)

    assert num_jobs == 2
    assert num_machines == 2
    assert max_ops_per_job == 2
    assert float(td["ref_makespan"].item()) == 11.0
    proc_times = td["proc_times"][0]
    assert float(proc_times[0, 0].item()) == 3.0
    assert float(proc_times[1, 1].item()) == 5.0
    assert float(proc_times[1, 2].item()) == 4.0
    assert float(proc_times[0, 3].item()) == 6.0


def test_jssp_file_generator_clips_requested_batch_to_available_files(tmp_path: Path) -> None:
    file_path = tmp_path / "toy.jsp"
    _write_bopo_like_jsp(file_path)

    generator = JSSPFileGenerator(tmp_path.as_posix())
    td = generator(batch_size=[8])

    assert tuple(td.batch_size) == (1,)
    assert float(td["ref_makespan"][0].item()) == 11.0
