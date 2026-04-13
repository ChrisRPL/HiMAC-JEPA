from pathlib import Path
import torch

from scripts.evaluate_baselines import (
    create_comparison_plots,
    create_comparison_table,
    evaluate_bev_probe,
    evaluate_motion_probe,
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
            "motion/ade": 0.8,
            "model/inference_time_ms": 8.5,
        },
    }
    per_sample_metrics = {
        "camera_only": {
            "trajectory/ade_3s": [1.4, 1.6, 1.5],
            "trajectory/fde_3s": [2.1, 2.4, 2.3],
            "motion/ade": [1.2, 1.1, 1.3],
        },
        "himac_jepa": {
            "trajectory/ade_3s": [1.0, 1.1, 1.2],
            "trajectory/fde_3s": [1.6, 1.8, 1.7],
            "motion/ade": [0.8, 0.9, 0.7],
        },
    }

    create_comparison_table(results, tmp_path)
    create_comparison_plots(results, tmp_path)
    run_statistical_tests(results, per_sample_metrics, tmp_path)

    assert (tmp_path / "metrics.csv").exists()
    assert (tmp_path / "per_sample_metrics.csv").exists()
    assert (tmp_path / "comparison_table.txt").exists()
    assert (tmp_path / "comparison_table.tex").exists()
    assert (tmp_path / "statistical_tests.txt").exists()
    assert (tmp_path / "plots" / "trajectory_ade.png").exists()
    assert (tmp_path / "plots" / "bev_miou.png").exists()
    assert (tmp_path / "plots" / "motion_ade.png").exists()


def test_statistical_tests_file_is_honest(tmp_path):
    run_statistical_tests({}, {}, tmp_path)

    contents = (tmp_path / "statistical_tests.txt").read_text()
    assert "skipped" in contents.lower()
    assert "aligned per-sample trajectory errors" in contents


def test_statistical_tests_report_paired_trajectory_results(tmp_path):
    results = {
        "camera_only": {
            "trajectory/ade_3s": 1.5,
            "trajectory/fde_3s": 2.3,
        },
        "himac_jepa": {
            "trajectory/ade_3s": 1.1,
            "trajectory/fde_3s": 1.7,
        },
    }
    per_sample_metrics = {
        "camera_only": {
            "trajectory/ade_3s": [1.6, 1.5, 1.4, 1.5],
            "trajectory/fde_3s": [2.5, 2.3, 2.1, 2.3],
        },
        "himac_jepa": {
            "trajectory/ade_3s": [1.2, 1.1, 1.0, 1.1],
            "trajectory/fde_3s": [1.9, 1.7, 1.5, 1.7],
        },
    }

    run_statistical_tests(results, per_sample_metrics, tmp_path)

    contents = (tmp_path / "statistical_tests.txt").read_text()
    assert "paired sign-flip permutation test" in contents.lower()
    assert "himac_jepa vs camera_only" in contents


def test_statistical_tests_report_motion_results_when_available(tmp_path):
    results = {
        "camera_only": {"motion/ade": 1.2},
        "himac_jepa": {"motion/ade": 0.8},
    }
    per_sample_metrics = {
        "camera_only": {"motion/ade": [1.3, 1.1, 1.2]},
        "himac_jepa": {"motion/ade": [0.9, 0.8, 0.7]},
    }

    run_statistical_tests(results, per_sample_metrics, tmp_path)

    contents = (tmp_path / "statistical_tests.txt").read_text()
    assert "motion/ade" in contents
    assert "himac_jepa vs camera_only" in contents


def test_evaluate_bev_probe_returns_metrics():
    train_probe_data = {
        "latents": torch.randn(4, 8),
        "bev_targets": torch.randint(0, 2, (4, 8, 8), dtype=torch.long),
    }
    val_probe_data = {
        "latents": torch.randn(2, 8),
        "bev_targets": torch.randint(0, 2, (2, 8, 8), dtype=torch.long),
    }

    metrics = evaluate_bev_probe(
        train_probe_data,
        val_probe_data,
        device=torch.device("cpu"),
        probe_epochs=1,
        probe_batch_size=2,
        probe_learning_rate=1e-3,
    )

    assert set(metrics) == {"bev/miou", "bev/precision", "bev/recall"}


def test_evaluate_motion_probe_returns_metrics():
    train_probe_data = {
        "latents": torch.randn(4, 8),
        "motion_targets": torch.randn(4, 2, 3, 2),
        "motion_valid_mask": torch.ones(4, 2, 3, dtype=torch.bool),
        "motion_agent_mask": torch.tensor(
            [[True, True], [True, False], [True, True], [True, False]]
        ),
    }
    val_probe_data = {
        "latents": torch.randn(2, 8),
        "motion_targets": torch.randn(2, 2, 3, 2),
        "motion_valid_mask": torch.ones(2, 2, 3, dtype=torch.bool),
        "motion_agent_mask": torch.tensor([[True, True], [True, False]]),
    }

    metrics, per_sample_metrics = evaluate_motion_probe(
        train_probe_data,
        val_probe_data,
        max_agents=2,
    )

    assert set(metrics) == {"motion/ade", "motion/fde"}
    assert set(per_sample_metrics) == {"motion/ade", "motion/fde"}
