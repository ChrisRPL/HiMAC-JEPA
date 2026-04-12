import torch

from src.evaluation.baseline_benchmark import (
    collect_probe_targets,
    compute_bev_classification_metrics,
    compute_trajectory_horizon_metrics,
    fit_ridge_probe,
    predict_ridge_probe,
)


def test_fit_ridge_probe_recovers_linear_mapping():
    train_x = torch.tensor([[1.0, 2.0], [2.0, 1.0], [3.0, 4.0], [4.0, 3.0]])
    train_y = torch.stack([
        2 * train_x[:, 0] - train_x[:, 1] + 1,
        -train_x[:, 0] + 3 * train_x[:, 1] - 2,
    ], dim=1)

    weights = fit_ridge_probe(train_x, train_y, alpha=1e-6)
    predicted = predict_ridge_probe(weights, train_x)

    assert torch.allclose(predicted, train_y, atol=1e-4)


def test_compute_trajectory_horizon_metrics_respects_valid_mask():
    predictions = torch.tensor(
        [
            [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0]],
            [[0.0, 0.0], [2.0, 0.0], [9.0, 9.0], [9.0, 9.0]],
        ]
    )
    targets = torch.tensor(
        [
            [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0]],
            [[0.0, 0.0], [1.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
        ]
    )
    valid_mask = torch.tensor(
        [
            [True, True, True, True],
            [True, True, False, False],
        ]
    )

    metrics = compute_trajectory_horizon_metrics(
        predictions,
        targets,
        valid_mask,
        sampling_rate=2.0,
        horizons=(1.0, 2.0),
    )

    assert metrics["trajectory/ade_1s"] == 0.25
    assert metrics["trajectory/fde_1s"] == 0.5
    assert metrics["trajectory/ade_2s"] == 0.1666666716337204
    assert metrics["trajectory/fde_2s"] == 0.5


def test_collect_probe_targets_reads_collated_fields():
    batch = {
        "trajectory_ego": torch.zeros(2, 6, 2),
        "trajectory_valid_mask": torch.ones(2, 6, dtype=torch.bool),
    }

    targets, valid_mask = collect_probe_targets(batch)

    assert targets.shape == (2, 6, 2)
    assert valid_mask.shape == (2, 6)


def test_compute_bev_classification_metrics():
    predictions = torch.tensor([[[0, 1], [1, 0]]])
    targets = torch.tensor([[[0, 1], [1, 0]]])

    metrics = compute_bev_classification_metrics(predictions, targets, num_classes=2)

    assert metrics["bev/miou"] == 1.0
    assert metrics["bev/precision"] == 1.0
    assert metrics["bev/recall"] == 1.0
