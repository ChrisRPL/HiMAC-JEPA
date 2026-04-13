import pytest
import torch

from src.evaluation.baseline_benchmark import (
    align_motion_targets,
    build_motion_probe_targets,
    collect_bev_targets,
    collect_motion_targets,
    collect_probe_targets,
    compute_bev_classification_metrics,
    compute_motion_metrics,
    compute_trajectory_horizon_errors,
    compute_trajectory_horizon_metrics,
    fit_bev_probe,
    fit_ridge_probe,
    paired_sign_flip_test,
    predict_bev_probe,
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


def test_compute_trajectory_horizon_errors_returns_per_sample_vectors():
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

    errors = compute_trajectory_horizon_errors(
        predictions,
        targets,
        valid_mask,
        sampling_rate=2.0,
        horizons=(1.0, 2.0),
    )

    assert torch.allclose(errors["trajectory/ade_1s"], torch.tensor([0.0, 0.5]))
    assert torch.allclose(errors["trajectory/fde_1s"], torch.tensor([0.0, 1.0]))
    assert torch.allclose(errors["trajectory/ade_2s"], torch.tensor([0.0, 0.5]))
    assert torch.allclose(errors["trajectory/fde_2s"], torch.tensor([0.0, 1.0]))


def test_collect_probe_targets_reads_collated_fields():
    batch = {
        "trajectory_ego": torch.zeros(2, 6, 2),
        "trajectory_valid_mask": torch.ones(2, 6, dtype=torch.bool),
    }

    targets, valid_mask = collect_probe_targets(batch)

    assert targets.shape == (2, 6, 2)
    assert valid_mask.shape == (2, 6)


def test_collect_bev_targets_reads_collated_field():
    batch = {
        "bev_label": torch.zeros(2, 8, 8, dtype=torch.long),
    }

    targets = collect_bev_targets(batch)

    assert targets.shape == (2, 8, 8)


def test_collect_motion_targets_reads_collated_fields():
    batch = {
        "motion_future_trajectories": torch.zeros(2, 3, 4, 2),
        "motion_valid_mask": torch.ones(2, 3, 4, dtype=torch.bool),
        "motion_agent_mask": torch.tensor([[True, True, False], [True, False, False]]),
    }

    trajectories, valid_mask, agent_mask = collect_motion_targets(batch)

    assert trajectories.shape == (2, 3, 4, 2)
    assert valid_mask.shape == (2, 3, 4)
    assert agent_mask.shape == (2, 3)


def test_compute_bev_classification_metrics():
    predictions = torch.tensor([[[0, 1], [1, 0]]])
    targets = torch.tensor([[[0, 1], [1, 0]]])

    metrics = compute_bev_classification_metrics(predictions, targets, num_classes=2)

    assert metrics["bev/miou"] == 1.0
    assert metrics["bev/precision"] == 1.0
    assert metrics["bev/recall"] == 1.0


def test_bev_probe_predicts_segmentation_shape():
    train_latents = torch.randn(4, 8)
    train_labels = torch.randint(0, 2, (4, 8, 8), dtype=torch.long)

    probe = fit_bev_probe(
        train_latents=train_latents,
        train_labels=train_labels,
        latent_dim=8,
        num_classes=2,
        bev_h=8,
        bev_w=8,
        device=torch.device("cpu"),
        epochs=1,
        batch_size=2,
        learning_rate=1e-3,
    )
    predictions = predict_bev_probe(
        probe,
        train_latents,
        device=torch.device("cpu"),
        batch_size=2,
    )

    assert predictions.shape == (4, 8, 8)


def test_build_motion_probe_targets_trims_to_shared_agent_budget():
    future = torch.arange(2 * 3 * 2 * 2, dtype=torch.float32).view(2, 3, 2, 2)
    valid = torch.tensor(
        [
            [[True, True], [True, False], [False, False]],
            [[True, True], [True, True], [True, False]],
        ]
    )
    agent_mask = torch.tensor([[True, True, False], [True, False, False]])

    targets, valid_mask, aligned_agents = build_motion_probe_targets(
        future,
        valid,
        agent_mask,
        max_agents=2,
    )

    assert targets.shape == (2, 2, 2, 2)
    assert valid_mask.shape == (2, 2, 2)
    assert aligned_agents.shape == (2, 2)
    assert torch.equal(aligned_agents[0], torch.tensor([True, True]))
    assert torch.equal(aligned_agents[1], torch.tensor([True, False]))


def test_compute_motion_metrics_respects_valid_masks():
    predictions = torch.tensor(
        [[
            [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]],
            [[0.0, 1.0], [0.0, 3.0], [9.0, 9.0]],
        ]]
    )
    targets = torch.tensor(
        [[
            [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]],
            [[0.0, 1.0], [0.0, 2.0], [0.0, 0.0]],
        ]]
    )
    valid_mask = torch.tensor([[[True, True, True], [True, True, False]]])
    agent_mask = torch.tensor([[True, True]])

    metrics = compute_motion_metrics(predictions, targets, valid_mask, agent_mask)

    assert metrics["motion/ade"] == pytest.approx(0.2)
    assert metrics["motion/fde"] == pytest.approx(0.5)


def test_align_motion_targets_reshapes_flat_predictions():
    prediction = torch.arange(12, dtype=torch.float32).view(1, 12)
    target = torch.zeros(1, 1, 3, 2)
    valid_mask = torch.ones(1, 1, 3, dtype=torch.bool)
    agent_mask = torch.ones(1, 1, dtype=torch.bool)

    aligned_prediction, aligned_target, aligned_valid_mask, aligned_agent_mask = align_motion_targets(
        prediction,
        target,
        valid_mask,
        agent_mask,
    )

    assert aligned_prediction.shape == (1, 1, 3, 2)
    assert aligned_target.shape == (1, 1, 3, 2)
    assert torch.equal(aligned_valid_mask, valid_mask)
    assert torch.equal(aligned_agent_mask, agent_mask)


def test_paired_sign_flip_test_detects_nonzero_delta():
    metric_a = torch.tensor([0.2, 0.3, 0.4, 0.5])
    metric_b = torch.tensor([0.4, 0.5, 0.6, 0.7])

    observed_delta, p_value, num_pairs = paired_sign_flip_test(
        metric_a,
        metric_b,
        num_permutations=2048,
        seed=7,
    )

    assert observed_delta < 0.0
    assert p_value < 0.2
    assert num_pairs == 4
