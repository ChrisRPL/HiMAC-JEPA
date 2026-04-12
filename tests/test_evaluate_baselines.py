from pathlib import Path

from scripts.evaluate_baselines import (
    create_comparison_plots,
    create_comparison_table,
    run_statistical_tests,
)


def test_comparison_artifacts_handle_missing_metrics(tmp_path):
    results = {
        "camera_only": {
            "trajectory/ade_1s": 0.8,
            "trajectory/ade_2s": 1.2,
            "trajectory/ade_3s": 1.5,
            "trajectory/fde_1s": 1.0,
            "trajectory/fde_2s": 1.8,
            "trajectory/fde_3s": 2.3,
            "model/inference_time_ms": 4.2,
        },
        "himac_jepa": {
            "trajectory/ade_3s": 1.1,
            "trajectory/fde_3s": 1.7,
            "bev/miou": 0.42,
            "model/inference_time_ms": 8.5,
        },
    }

    create_comparison_table(results, tmp_path)
    create_comparison_plots(results, tmp_path)
    run_statistical_tests(results, tmp_path)

    assert (tmp_path / "metrics.csv").exists()
    assert (tmp_path / "comparison_table.txt").exists()
    assert (tmp_path / "comparison_table.tex").exists()
    assert (tmp_path / "statistical_tests.txt").exists()
    assert (tmp_path / "plots" / "trajectory_ade.png").exists()
    assert (tmp_path / "plots" / "bev_miou.png").exists()


def test_statistical_tests_file_is_honest(tmp_path):
    run_statistical_tests({}, tmp_path)

    contents = (tmp_path / "statistical_tests.txt").read_text()
    assert "skipped" in contents.lower()
    assert "paired per-sample outputs" in contents
