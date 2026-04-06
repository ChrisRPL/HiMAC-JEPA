"""Tests for evaluation metrics."""
import pytest
import torch
import numpy as np


class _DummyEvalModel:
    def eval(self):
        return self


class _ConstantPredictionModel:
    def __init__(self, trajectory, bev_logits):
        self.trajectory = trajectory
        self.bev_logits = bev_logits

    def eval(self):
        return self

    def __call__(self, camera, lidar, radar, strategic, tactical, masks=None):
        batch_size = camera.shape[0]
        trajectory = self.trajectory[:batch_size].clone()
        bev_logits = self.bev_logits[:batch_size].clone()
        latent = torch.zeros(batch_size, 4)
        log_var = torch.zeros(batch_size, 4)
        motion = torch.zeros(batch_size, 6)
        return latent, log_var, trajectory, motion, bev_logits


class _TemporalOnlyModel:
    def eval(self):
        return self

    def encode_observations(self, camera, lidar, radar, masks=None):
        pooled = camera.mean(dim=(1, 2, 3))
        return torch.stack([pooled, torch.zeros_like(pooled)], dim=1)


class TestIntrinsicMetrics:
    """Test intrinsic evaluation metrics."""

    def test_latent_mse_computation(self):
        """Test latent MSE calculation."""
        pytest.importorskip("sklearn")
        from src.evaluation.intrinsic_metrics import IntrinsicEvaluator

        # Create mock model and dataloader
        # TODO: Implement with proper mocks
        assert True  # Placeholder

    def test_linear_probe(self):
        """Test linear probing functionality."""
        pytest.importorskip("sklearn")
        from src.evaluation.intrinsic_metrics import IntrinsicEvaluator

        # TODO: Test linear probe with dummy embeddings
        assert True  # Placeholder

    def test_silhouette_score(self):
        """Test embedding silhouette score computation."""
        pytest.importorskip("sklearn")
        from src.evaluation.intrinsic_metrics import IntrinsicEvaluator

        # TODO: Test with known embeddings
        assert True  # Placeholder

    def test_run_all_skips_latent_mse_without_teacher(self):
        from types import SimpleNamespace
        from src.evaluation.intrinsic_metrics import IntrinsicEvaluator

        config = SimpleNamespace(
            metrics=SimpleNamespace(intrinsic=["latent_mse"]),
            intrinsic=SimpleNamespace(
                linear_probe_tasks=[],
                num_probe_epochs=1,
                probe_learning_rate=0.001,
                compute_temporal_consistency=False,
                temporal_window=3,
            ),
        )

        evaluator = IntrinsicEvaluator(_DummyEvalModel(), [], "cpu")
        results = evaluator.run_all(config)

        assert "intrinsic/latent_mse" not in results

    def test_temporal_consistency_uses_temporal_batches(self):
        from src.evaluation.intrinsic_metrics import IntrinsicEvaluator

        batch = {
            "context": {
                "camera": torch.tensor(
                    [[
                        [[[0.0, 0.0], [0.0, 0.0]]],
                        [[[1.0, 1.0], [1.0, 1.0]]],
                        [[[3.0, 3.0], [3.0, 3.0]]],
                    ]],
                    dtype=torch.float32,
                ),
                "lidar": torch.zeros(1, 3, 2, 3),
                "radar": torch.zeros(1, 3, 1, 2, 2),
            }
        }

        evaluator = IntrinsicEvaluator(_TemporalOnlyModel(), [batch], "cpu")
        score = evaluator.temporal_consistency()

        assert score == pytest.approx(1.5)


class TestDownstreamMetrics:
    """Test downstream evaluation metrics."""

    def test_ade_calculation(self):
        """Test ADE (Average Displacement Error) calculation."""
        from src.evaluation.downstream_metrics import DownstreamEvaluator

        evaluator = DownstreamEvaluator(_DummyEvalModel(), None, 'cpu')

        # Create dummy trajectories
        pred = torch.randn(2, 10, 2)  # (batch=2, time=10, coords=2)
        gt = torch.randn(2, 10, 2)

        ade = evaluator._compute_ade(pred, gt)

        assert isinstance(ade, float)
        assert ade >= 0

    def test_fde_calculation(self):
        """Test FDE (Final Displacement Error) calculation."""
        from src.evaluation.downstream_metrics import DownstreamEvaluator

        evaluator = DownstreamEvaluator(_DummyEvalModel(), None, 'cpu')

        # Create dummy trajectories
        pred = torch.randn(2, 10, 2)
        gt = torch.randn(2, 10, 2)

        fde = evaluator._compute_fde(pred, gt)

        assert isinstance(fde, float)
        assert fde >= 0

    def test_fde_perfect_prediction(self):
        """Test FDE with perfect prediction (should be 0)."""
        from src.evaluation.downstream_metrics import DownstreamEvaluator

        evaluator = DownstreamEvaluator(_DummyEvalModel(), None, 'cpu')

        # Perfect prediction
        pred = torch.randn(2, 10, 2)
        gt = pred.clone()

        fde = evaluator._compute_fde(pred, gt)

        assert fde < 1e-6  # Should be very close to 0

    def test_iou_calculation(self):
        """Test IoU calculation for segmentation."""
        from src.evaluation.downstream_metrics import DownstreamEvaluator

        evaluator = DownstreamEvaluator(_DummyEvalModel(), None, 'cpu')

        # Create dummy segmentation masks
        pred = np.array([[0, 0, 1], [1, 1, 0], [0, 1, 1]])
        gt = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 1]])

        iou = evaluator._compute_iou(pred, gt, num_classes=2)

        assert isinstance(iou, np.ndarray)
        assert len(iou) == 2
        assert np.all((iou >= 0) | np.isnan(iou))
        assert np.all((iou <= 1) | np.isnan(iou))

    def test_iou_perfect_match(self):
        """Test IoU with perfect segmentation match."""
        from src.evaluation.downstream_metrics import DownstreamEvaluator

        evaluator = DownstreamEvaluator(_DummyEvalModel(), None, 'cpu')

        # Perfect match
        pred = np.array([[0, 1, 1], [1, 0, 0]])
        gt = pred.copy()

        iou = evaluator._compute_iou(pred, gt, num_classes=2)

        # Filter out NaN values (classes not present)
        valid_iou = iou[~np.isnan(iou)]
        assert np.all(np.abs(valid_iou - 1.0) < 1e-6)  # Should be 1.0

    def test_iou_no_match(self):
        """Test IoU with no overlap."""
        from src.evaluation.downstream_metrics import DownstreamEvaluator

        evaluator = DownstreamEvaluator(_DummyEvalModel(), None, 'cpu')

        # No overlap
        pred = np.array([[0, 0, 0], [0, 0, 0]])
        gt = np.array([[1, 1, 1], [1, 1, 1]])

        iou = evaluator._compute_iou(pred, gt, num_classes=2)

        # Class 0 and class 1 should have 0 IoU
        assert iou[0] < 1e-6 or np.isnan(iou[0])
        assert iou[1] < 1e-6 or np.isnan(iou[1])

    def test_confusion_matrix(self):
        """Test confusion matrix computation."""
        from src.evaluation.downstream_metrics import DownstreamEvaluator

        evaluator = DownstreamEvaluator(_DummyEvalModel(), None, 'cpu')

        # Create dummy predictions
        pred = np.array([[0, 1, 1], [1, 0, 1]])
        gt = np.array([[0, 1, 0], [1, 0, 0]])

        tp, fp, fn = evaluator._compute_confusion_matrix(pred, gt)

        assert isinstance(tp, int)
        assert isinstance(fp, int)
        assert isinstance(fn, int)
        assert tp >= 0
        assert fp >= 0
        assert fn >= 0

    def test_trajectory_metrics_uses_batch_targets(self):
        from src.evaluation.downstream_metrics import DownstreamEvaluator

        batch = {
            "camera": torch.zeros(2, 3, 4, 4),
            "lidar": torch.zeros(2, 8, 3),
            "radar": torch.zeros(2, 1, 4, 4),
            "strategic_action": torch.zeros(2, dtype=torch.long),
            "tactical_action": torch.zeros(2, 3),
            "trajectory_ego": torch.tensor(
                [
                    [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]],
                    [[0.0, 0.0], [1.0, 1.0], [0.0, 0.0]],
                ],
                dtype=torch.float32,
            ),
            "trajectory_valid_mask": torch.tensor(
                [[True, True, True], [True, True, False]]
            ),
        }

        trajectory_pred = torch.tensor(
            [
                [0.0, 0.0, 1.0, 0.0, 2.0, 0.0],
                [0.0, 0.0, 2.0, 1.0, 9.0, 9.0],
            ],
            dtype=torch.float32,
        )
        bev_logits = torch.zeros(2, 2, 2, 2)

        evaluator = DownstreamEvaluator(
            _ConstantPredictionModel(trajectory_pred, bev_logits),
            [batch],
            "cpu",
        )

        metrics = evaluator.trajectory_metrics()

        assert metrics["downstream/trajectory_ade"] == pytest.approx(0.2)
        assert metrics["downstream/trajectory_fde"] == pytest.approx(0.5)
        assert metrics["downstream/trajectory_min_ade"] == pytest.approx(0.2)
        assert metrics["downstream/trajectory_min_fde"] == pytest.approx(0.5)

    def test_bev_segmentation_metrics_uses_batch_targets(self):
        from src.evaluation.downstream_metrics import DownstreamEvaluator

        batch = {
            "camera": torch.zeros(1, 3, 4, 4),
            "lidar": torch.zeros(1, 8, 3),
            "radar": torch.zeros(1, 1, 4, 4),
            "strategic_action": torch.zeros(1, dtype=torch.long),
            "tactical_action": torch.zeros(1, 3),
            "bev_label": torch.tensor([[[0, 1], [1, 0]]], dtype=torch.long),
        }

        bev_logits = torch.tensor(
            [[
                [[4.0, 1.0], [1.0, 3.0]],
                [[1.0, 4.0], [4.0, 1.0]],
            ]],
            dtype=torch.float32,
        )
        trajectory_pred = torch.zeros(1, 6)

        evaluator = DownstreamEvaluator(
            _ConstantPredictionModel(trajectory_pred, bev_logits),
            [batch],
            "cpu",
        )

        metrics = evaluator.bev_segmentation_metrics(num_classes=2)

        assert metrics["downstream/bev_miou"] == pytest.approx(1.0)
        assert metrics["downstream/bev_precision"] == pytest.approx(1.0)
        assert metrics["downstream/bev_recall"] == pytest.approx(1.0)


class TestMetricsEdgeCases:
    """Test edge cases for metrics."""

    def test_empty_predictions(self):
        """Test metrics with empty predictions."""
        from src.evaluation.downstream_metrics import DownstreamEvaluator

        evaluator = DownstreamEvaluator(_DummyEvalModel(), None, 'cpu')

        # Empty predictions
        pred = torch.zeros(0, 10, 2)
        gt = torch.zeros(0, 10, 2)

        # Should handle gracefully (not crash)
        try:
            ade = evaluator._compute_ade(pred, gt)
            assert True  # Passed if no exception
        except:
            pytest.fail("Should handle empty predictions")

    def test_nan_handling(self):
        """Test that NaN values are handled correctly."""
        from src.evaluation.downstream_metrics import DownstreamEvaluator

        evaluator = DownstreamEvaluator(_DummyEvalModel(), None, 'cpu')

        # Predictions with NaN
        pred = torch.tensor([[[1.0, 2.0], [float('nan'), 3.0]]])
        gt = torch.tensor([[[1.0, 2.0], [2.0, 3.0]]])

        # Should detect or handle NaN
        ade = evaluator._compute_ade(pred, gt)

        # Either NaN or raises warning
        assert isinstance(ade, float)


class TestEvaluationBatching:
    """Test evaluation-specific batch collation helpers."""

    def test_collate_evaluation_batch_pads_trajectory_targets(self):
        from src.evaluation.batching import collate_evaluation_batch

        samples = [
            {
                "camera": torch.zeros(3, 4, 4),
                "lidar": torch.zeros(8, 3),
                "radar": torch.zeros(1, 4, 4),
                "strategic_action": torch.tensor(1),
                "tactical_action": torch.zeros(3),
                "labels": {
                    "trajectory_ego": {
                        1.0: np.array([[1.0, 0.0], [2.0, 0.0]], dtype=np.float32),
                        3.0: np.array([[1.0, 0.0], [2.0, 0.0], [3.0, 0.0]], dtype=np.float32),
                    },
                    "bev": np.ones((2, 2), dtype=np.uint8),
                },
            },
            {
                "camera": torch.ones(3, 4, 4),
                "lidar": torch.ones(8, 3),
                "radar": torch.ones(1, 4, 4),
                "strategic_action": torch.tensor(2),
                "tactical_action": torch.ones(3),
                "labels": {
                    "trajectory_ego": {
                        1.0: np.array([[0.5, 0.5]], dtype=np.float32),
                        3.0: np.array([[0.5, 0.5], [1.0, 1.0]], dtype=np.float32),
                    },
                    "bev": np.zeros((2, 2), dtype=np.uint8),
                },
            },
        ]

        batch = collate_evaluation_batch(samples)

        assert batch["trajectory_ego"].shape == (2, 3, 2)
        assert batch["trajectory_valid_mask"].shape == (2, 3)
        assert torch.equal(batch["trajectory_valid_mask"][0], torch.tensor([True, True, True]))
        assert torch.equal(batch["trajectory_valid_mask"][1], torch.tensor([True, True, False]))
        assert batch["bev_label"].shape == (2, 2, 2)
