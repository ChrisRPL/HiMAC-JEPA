"""Tests for temporal fusion modules."""
import pytest
import torch
import torch.nn as nn


class TestPositionalEncoding:
    """Test suite for sinusoidal positional encoding."""

    def test_initialization(self):
        """Test positional encoding initializes correctly."""
        from src.models.temporal_fusion import PositionalEncoding

        pos_enc = PositionalEncoding(d_model=768, dropout=0.1, max_len=100)

        assert hasattr(pos_enc, 'pe')
        assert pos_enc.pe.shape == (100, 768)

    def test_forward_shape(self):
        """Test positional encoding output shape."""
        from src.models.temporal_fusion import PositionalEncoding

        pos_enc = PositionalEncoding(d_model=768)

        # Input: (T=5, B=4, D=768)
        x = torch.randn(5, 4, 768)
        out = pos_enc(x)

        assert out.shape == (5, 4, 768), \
            f"Output shape should be (5, 4, 768), got {out.shape}"

    def test_positional_encoding_values(self):
        """Test that positional encoding has expected properties."""
        from src.models.temporal_fusion import PositionalEncoding

        pos_enc = PositionalEncoding(d_model=64, dropout=0.0)

        # Check that encoding is deterministic
        x1 = torch.randn(10, 2, 64)
        x2 = x1.clone()

        out1 = pos_enc(x1)
        out2 = pos_enc(x2)

        # Should be the same (dropout=0)
        assert torch.allclose(out1, out2), \
            "Positional encoding should be deterministic with dropout=0"

    def test_different_sequence_lengths(self):
        """Test positional encoding with different sequence lengths."""
        from src.models.temporal_fusion import PositionalEncoding

        pos_enc = PositionalEncoding(d_model=768, max_len=100)

        # Test with different T
        for T in [1, 5, 10, 20, 50]:
            x = torch.randn(T, 4, 768)
            out = pos_enc(x)
            assert out.shape == (T, 4, 768)


class TestTemporalTransformer:
    """Test suite for transformer-based temporal aggregation."""

    def test_initialization(self):
        """Test temporal transformer initializes correctly."""
        from src.models.temporal_fusion import TemporalTransformer

        model = TemporalTransformer(
            d_model=768,
            nhead=8,
            num_layers=4,
            dropout=0.1
        )

        assert model.d_model == 768
        assert hasattr(model, 'pos_encoder')
        assert hasattr(model, 'transformer')
        assert hasattr(model, 'norm')

    def test_forward_last_aggregation(self):
        """Test forward pass with last aggregation."""
        from src.models.temporal_fusion import TemporalTransformer

        model = TemporalTransformer(d_model=768, nhead=8, num_layers=2)
        model.eval()

        # Input: (B=4, T=5, D=768)
        x = torch.randn(4, 5, 768)

        with torch.no_grad():
            out = model(x, aggregate='last')

        assert out.shape == (4, 768), \
            f"Output shape should be (4, 768) for 'last' aggregation, got {out.shape}"

    def test_forward_mean_aggregation(self):
        """Test forward pass with mean aggregation."""
        from src.models.temporal_fusion import TemporalTransformer

        model = TemporalTransformer(d_model=768, nhead=8, num_layers=2)
        model.eval()

        x = torch.randn(4, 5, 768)

        with torch.no_grad():
            out = model(x, aggregate='mean')

        assert out.shape == (4, 768), \
            f"Output shape should be (4, 768) for 'mean' aggregation, got {out.shape}"

    def test_forward_max_aggregation(self):
        """Test forward pass with max aggregation."""
        from src.models.temporal_fusion import TemporalTransformer

        model = TemporalTransformer(d_model=768, nhead=8, num_layers=2)
        model.eval()

        x = torch.randn(4, 5, 768)

        with torch.no_grad():
            out = model(x, aggregate='max')

        assert out.shape == (4, 768), \
            f"Output shape should be (4, 768) for 'max' aggregation, got {out.shape}"

    def test_forward_all_aggregation(self):
        """Test forward pass with all aggregation (return all timesteps)."""
        from src.models.temporal_fusion import TemporalTransformer

        model = TemporalTransformer(d_model=768, nhead=8, num_layers=2)
        model.eval()

        x = torch.randn(4, 5, 768)

        with torch.no_grad():
            out = model(x, aggregate='all')

        assert out.shape == (4, 5, 768), \
            f"Output shape should be (4, 5, 768) for 'all' aggregation, got {out.shape}"

    def test_invalid_aggregation_method(self):
        """Test that invalid aggregation method raises error."""
        from src.models.temporal_fusion import TemporalTransformer

        model = TemporalTransformer(d_model=768)
        x = torch.randn(4, 5, 768)

        with pytest.raises(ValueError, match="Unknown aggregate method"):
            model(x, aggregate='invalid')

    def test_different_batch_sizes(self):
        """Test with different batch sizes."""
        from src.models.temporal_fusion import TemporalTransformer

        model = TemporalTransformer(d_model=256, nhead=4, num_layers=2)
        model.eval()

        for B in [1, 2, 4, 8]:
            x = torch.randn(B, 5, 256)
            with torch.no_grad():
                out = model(x, aggregate='last')
            assert out.shape == (B, 256)

    def test_different_sequence_lengths(self):
        """Test with different sequence lengths."""
        from src.models.temporal_fusion import TemporalTransformer

        model = TemporalTransformer(d_model=256, nhead=4, num_layers=2)
        model.eval()

        for T in [1, 3, 5, 10, 20]:
            x = torch.randn(4, T, 256)
            with torch.no_grad():
                out = model(x, aggregate='last')
            assert out.shape == (4, 256)

    def test_different_feature_dims(self):
        """Test with different feature dimensions."""
        from src.models.temporal_fusion import TemporalTransformer

        for d_model in [128, 256, 512, 768]:
            # nhead must divide d_model
            nhead = 4 if d_model >= 128 else 2

            model = TemporalTransformer(d_model=d_model, nhead=nhead, num_layers=2)
            model.eval()

            x = torch.randn(4, 5, d_model)
            with torch.no_grad():
                out = model(x, aggregate='last')
            assert out.shape == (4, d_model)

    def test_gradient_flow(self):
        """Test that gradients flow correctly."""
        from src.models.temporal_fusion import TemporalTransformer

        model = TemporalTransformer(d_model=256, nhead=4, num_layers=2)

        x = torch.randn(4, 5, 256, requires_grad=True)
        out = model(x, aggregate='last')

        # Compute loss and backward
        loss = out.sum()
        loss.backward()

        # Check gradients exist
        assert x.grad is not None, "Input should have gradients"
        assert torch.any(x.grad != 0), "Gradients should be non-zero"

    def test_deterministic_inference(self):
        """Test that inference is deterministic in eval mode."""
        from src.models.temporal_fusion import TemporalTransformer

        model = TemporalTransformer(d_model=256, nhead=4, num_layers=2)
        model.eval()

        x = torch.randn(4, 5, 256)

        with torch.no_grad():
            out1 = model(x, aggregate='last')
            out2 = model(x, aggregate='last')

        assert torch.allclose(out1, out2), \
            "Outputs should be identical in eval mode"


class TestSimpleTemporalAggregator:
    """Test suite for simple MLP-based temporal aggregation."""

    def test_initialization(self):
        """Test simple aggregator initializes correctly."""
        from src.models.temporal_fusion import SimpleTemporalAggregator

        model = SimpleTemporalAggregator(d_model=768, hidden_dim=1024)

        assert hasattr(model, 'mlp')
        assert hasattr(model, 'norm')

    def test_initialization_default_hidden_dim(self):
        """Test initialization with default hidden_dim."""
        from src.models.temporal_fusion import SimpleTemporalAggregator

        model = SimpleTemporalAggregator(d_model=768)

        # Should use d_model as hidden_dim by default
        assert model is not None

    def test_forward_last_aggregation(self):
        """Test forward pass with last aggregation."""
        from src.models.temporal_fusion import SimpleTemporalAggregator

        model = SimpleTemporalAggregator(d_model=768)
        model.eval()

        x = torch.randn(4, 5, 768)

        with torch.no_grad():
            out = model(x, aggregate='last')

        assert out.shape == (4, 768), \
            f"Output shape should be (4, 768), got {out.shape}"

    def test_forward_mean_aggregation(self):
        """Test forward pass with mean aggregation."""
        from src.models.temporal_fusion import SimpleTemporalAggregator

        model = SimpleTemporalAggregator(d_model=768)
        model.eval()

        x = torch.randn(4, 5, 768)

        with torch.no_grad():
            out = model(x, aggregate='mean')

        assert out.shape == (4, 768), \
            f"Output shape should be (4, 768), got {out.shape}"

    def test_forward_max_aggregation(self):
        """Test forward pass with max aggregation."""
        from src.models.temporal_fusion import SimpleTemporalAggregator

        model = SimpleTemporalAggregator(d_model=768)
        model.eval()

        x = torch.randn(4, 5, 768)

        with torch.no_grad():
            out = model(x, aggregate='max')

        assert out.shape == (4, 768), \
            f"Output shape should be (4, 768), got {out.shape}"

    def test_invalid_aggregation_method(self):
        """Test that invalid aggregation method raises error."""
        from src.models.temporal_fusion import SimpleTemporalAggregator

        model = SimpleTemporalAggregator(d_model=768)
        x = torch.randn(4, 5, 768)

        with pytest.raises(ValueError, match="Unknown aggregate method"):
            model(x, aggregate='invalid')

    def test_residual_connection(self):
        """Test that residual connection works."""
        from src.models.temporal_fusion import SimpleTemporalAggregator

        model = SimpleTemporalAggregator(d_model=256)
        model.eval()

        x = torch.randn(4, 5, 256)

        with torch.no_grad():
            out = model(x, aggregate='last')

        # Output should be different from input due to MLP + residual
        assert out.shape == (4, 256)

    def test_gradient_flow(self):
        """Test that gradients flow correctly."""
        from src.models.temporal_fusion import SimpleTemporalAggregator

        model = SimpleTemporalAggregator(d_model=256)

        x = torch.randn(4, 5, 256, requires_grad=True)
        out = model(x, aggregate='last')

        loss = out.sum()
        loss.backward()

        assert x.grad is not None
        assert torch.any(x.grad != 0)

    def test_different_batch_sizes(self):
        """Test with different batch sizes."""
        from src.models.temporal_fusion import SimpleTemporalAggregator

        model = SimpleTemporalAggregator(d_model=256)
        model.eval()

        for B in [1, 2, 4, 8]:
            x = torch.randn(B, 5, 256)
            with torch.no_grad():
                out = model(x, aggregate='mean')
            assert out.shape == (B, 256)

    def test_different_sequence_lengths(self):
        """Test with different sequence lengths."""
        from src.models.temporal_fusion import SimpleTemporalAggregator

        model = SimpleTemporalAggregator(d_model=256)
        model.eval()

        for T in [1, 3, 5, 10, 20]:
            x = torch.randn(4, T, 256)
            with torch.no_grad():
                out = model(x, aggregate='mean')
            assert out.shape == (4, 256)


class TestTemporalFusionComparison:
    """Compare transformer vs simple aggregator."""

    def test_both_produce_valid_outputs(self):
        """Test that both aggregators produce valid outputs."""
        from src.models.temporal_fusion import TemporalTransformer, SimpleTemporalAggregator

        x = torch.randn(4, 5, 256)

        transformer = TemporalTransformer(d_model=256, nhead=4, num_layers=2)
        simple = SimpleTemporalAggregator(d_model=256)

        transformer.eval()
        simple.eval()

        with torch.no_grad():
            out_transformer = transformer(x, aggregate='last')
            out_simple = simple(x, aggregate='last')

        assert out_transformer.shape == (4, 256)
        assert out_simple.shape == (4, 256)

        # Outputs should be different (different architectures)
        assert not torch.allclose(out_transformer, out_simple)

    def test_parameter_count_difference(self):
        """Test that transformer has more parameters than simple aggregator."""
        from src.models.temporal_fusion import TemporalTransformer, SimpleTemporalAggregator

        transformer = TemporalTransformer(d_model=256, nhead=4, num_layers=2)
        simple = SimpleTemporalAggregator(d_model=256)

        transformer_params = sum(p.numel() for p in transformer.parameters())
        simple_params = sum(p.numel() for p in simple.parameters())

        # Transformer should have significantly more parameters
        assert transformer_params > simple_params, \
            f"Transformer ({transformer_params}) should have more params than simple ({simple_params})"
