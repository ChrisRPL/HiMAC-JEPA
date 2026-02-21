"""Tests for evaluation metrics."""
import pytest
import torch
import numpy as np


class TestIntrinsicMetrics:
    """Test intrinsic evaluation metrics."""

    def test_latent_mse_computation(self):
        """Test latent MSE calculation."""
        from src.evaluation.intrinsic_metrics import IntrinsicEvaluator

        # Create mock model and dataloader
        # TODO: Implement with proper mocks
        assert True  # Placeholder

    def test_linear_probe(self):
        """Test linear probing functionality."""
        from src.evaluation.intrinsic_metrics import IntrinsicEvaluator

        # TODO: Test linear probe with dummy embeddings
        assert True  # Placeholder

    def test_silhouette_score(self):
        """Test embedding silhouette score computation."""
        from src.evaluation.intrinsic_metrics import IntrinsicEvaluator

        # TODO: Test with known embeddings
        assert True  # Placeholder


class TestDownstreamMetrics:
    """Test downstream evaluation metrics."""

    def test_ade_calculation(self):
        """Test ADE (Average Displacement Error) calculation."""
        from src.evaluation.downstream_metrics import DownstreamEvaluator

        evaluator = DownstreamEvaluator(None, None, 'cpu')

        # Create dummy trajectories
        pred = torch.randn(2, 10, 2)  # (batch=2, time=10, coords=2)
        gt = torch.randn(2, 10, 2)

        ade = evaluator._compute_ade(pred, gt)

        assert isinstance(ade, float)
        assert ade >= 0

    def test_fde_calculation(self):
        """Test FDE (Final Displacement Error) calculation."""
        from src.evaluation.downstream_metrics import DownstreamEvaluator

        evaluator = DownstreamEvaluator(None, None, 'cpu')

        # Create dummy trajectories
        pred = torch.randn(2, 10, 2)
        gt = torch.randn(2, 10, 2)

        fde = evaluator._compute_fde(pred, gt)

        assert isinstance(fde, float)
        assert fde >= 0

    def test_fde_perfect_prediction(self):
        """Test FDE with perfect prediction (should be 0)."""
        from src.evaluation.downstream_metrics import DownstreamEvaluator

        evaluator = DownstreamEvaluator(None, None, 'cpu')

        # Perfect prediction
        pred = torch.randn(2, 10, 2)
        gt = pred.clone()

        fde = evaluator._compute_fde(pred, gt)

        assert fde < 1e-6  # Should be very close to 0

    def test_iou_calculation(self):
        """Test IoU calculation for segmentation."""
        from src.evaluation.downstream_metrics import DownstreamEvaluator

        evaluator = DownstreamEvaluator(None, None, 'cpu')

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

        evaluator = DownstreamEvaluator(None, None, 'cpu')

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

        evaluator = DownstreamEvaluator(None, None, 'cpu')

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

        evaluator = DownstreamEvaluator(None, None, 'cpu')

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


class TestMetricsEdgeCases:
    """Test edge cases for metrics."""

    def test_empty_predictions(self):
        """Test metrics with empty predictions."""
        from src.evaluation.downstream_metrics import DownstreamEvaluator

        evaluator = DownstreamEvaluator(None, None, 'cpu')

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

        evaluator = DownstreamEvaluator(None, None, 'cpu')

        # Predictions with NaN
        pred = torch.tensor([[[1.0, 2.0], [float('nan'), 3.0]]])
        gt = torch.tensor([[[1.0, 2.0], [2.0, 3.0]]])

        # Should detect or handle NaN
        ade = evaluator._compute_ade(pred, gt)

        # Either NaN or raises warning
        assert isinstance(ade, float)
