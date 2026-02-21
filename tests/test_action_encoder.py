import torch
import pytest
from src.models.action_encoder import HierarchicalActionEncoder


class TestHierarchicalActionEncoder:
    """Test suite for the HierarchicalActionEncoder module."""

    @pytest.fixture
    def encoder(self):
        """Create a default action encoder for testing."""
        return HierarchicalActionEncoder(
            strategic_vocab_size=10,
            tactical_dim=3,
            latent_dim=128,
            num_heads=8,
            depth=2
        )

    def test_initialization(self, encoder):
        """Test that the encoder initializes correctly."""
        assert encoder.strategic_vocab_size == 10
        assert encoder.tactical_dim == 3
        assert encoder.latent_dim == 128
        assert isinstance(encoder.strategic_embedding, torch.nn.Embedding)
        assert isinstance(encoder.tactical_mlp, torch.nn.Sequential)
        assert isinstance(encoder.fusion_transformer, torch.nn.TransformerEncoder)

    def test_forward_shape_1d_strategic(self, encoder):
        """Test forward pass with 1D strategic action tensor."""
        batch_size = 4
        strategic_action = torch.randint(0, 10, (batch_size,))
        tactical_action = torch.randn(batch_size, 3)

        output = encoder(strategic_action, tactical_action)

        assert output.shape == (batch_size, 128), f"Expected shape ({batch_size}, 128), got {output.shape}"

    def test_forward_shape_2d_strategic(self, encoder):
        """Test forward pass with 2D strategic action tensor (B, 1)."""
        batch_size = 4
        strategic_action = torch.randint(0, 10, (batch_size, 1))
        tactical_action = torch.randn(batch_size, 3)

        output = encoder(strategic_action, tactical_action)

        assert output.shape == (batch_size, 128), f"Expected shape ({batch_size}, 128), got {output.shape}"

    def test_strategic_embedding_lookup(self, encoder):
        """Test that strategic actions are properly embedded."""
        strategic_action = torch.tensor([0, 1, 2, 3])
        tactical_action = torch.randn(4, 3)

        # Get embeddings directly
        expected_embed = encoder.strategic_embedding(strategic_action)
        assert expected_embed.shape == (4, 64), "Strategic embedding shape mismatch"

        # Verify different actions get different embeddings
        embed_0 = encoder.strategic_embedding(torch.tensor([0]))
        embed_1 = encoder.strategic_embedding(torch.tensor([1]))
        assert not torch.allclose(embed_0, embed_1), "Different strategic actions should have different embeddings"

    def test_tactical_mlp_processing(self, encoder):
        """Test that tactical actions are processed through MLP."""
        tactical_action = torch.randn(4, 3)

        tactical_embed = encoder.tactical_mlp(tactical_action)
        assert tactical_embed.shape == (4, 64), "Tactical MLP output shape mismatch"

    def test_output_consistency(self, encoder):
        """Test that the same inputs produce the same outputs (deterministic)."""
        encoder.eval()  # Set to eval mode to disable dropout

        strategic_action = torch.tensor([1, 2, 3, 4])
        tactical_action = torch.randn(4, 3)

        with torch.no_grad():
            output1 = encoder(strategic_action, tactical_action)
            output2 = encoder(strategic_action, tactical_action)

        assert torch.allclose(output1, output2), "Encoder should be deterministic in eval mode"

    def test_gradient_flow(self, encoder):
        """Test that gradients flow correctly through the encoder."""
        encoder.train()

        strategic_action = torch.randint(0, 10, (4,))
        tactical_action = torch.randn(4, 3)

        output = encoder(strategic_action, tactical_action)
        loss = output.sum()
        loss.backward()

        # Check that gradients exist for key parameters
        assert encoder.strategic_embedding.weight.grad is not None, "Strategic embedding should have gradients"
        assert encoder.tactical_mlp[0].weight.grad is not None, "Tactical MLP should have gradients"
        assert encoder.output_proj.weight.grad is not None, "Output projection should have gradients"

    def test_batch_size_variations(self, encoder):
        """Test encoder with different batch sizes."""
        for batch_size in [1, 2, 8, 16, 32]:
            strategic_action = torch.randint(0, 10, (batch_size,))
            tactical_action = torch.randn(batch_size, 3)

            output = encoder(strategic_action, tactical_action)
            assert output.shape == (batch_size, 128), f"Failed for batch size {batch_size}"

    def test_different_latent_dims(self):
        """Test encoder with different latent dimensions."""
        for latent_dim in [64, 128, 256, 512]:
            encoder = HierarchicalActionEncoder(
                strategic_vocab_size=10,
                tactical_dim=3,
                latent_dim=latent_dim,
                num_heads=8,
                depth=2
            )

            strategic_action = torch.randint(0, 10, (4,))
            tactical_action = torch.randn(4, 3)

            output = encoder(strategic_action, tactical_action)
            assert output.shape == (4, latent_dim), f"Failed for latent_dim {latent_dim}"

    def test_invalid_strategic_action(self, encoder):
        """Test that invalid strategic action indices raise errors."""
        strategic_action = torch.tensor([10, 11])  # Out of vocab range [0, 9]
        tactical_action = torch.randn(2, 3)

        with pytest.raises(IndexError):
            encoder(strategic_action, tactical_action)

    def test_output_not_nan(self, encoder):
        """Test that output does not contain NaN values."""
        encoder.eval()

        strategic_action = torch.randint(0, 10, (4,))
        tactical_action = torch.randn(4, 3)

        with torch.no_grad():
            output = encoder(strategic_action, tactical_action)

        assert not torch.isnan(output).any(), "Output contains NaN values"
        assert not torch.isinf(output).any(), "Output contains Inf values"
