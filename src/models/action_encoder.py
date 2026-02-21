import torch
import torch.nn as nn


class HierarchicalActionEncoder(nn.Module):
    """Encodes hierarchical actions (strategic + tactical) into a latent representation.

    This encoder combines categorical strategic actions (e.g., "change lane", "keep lane")
    with continuous tactical actions (e.g., steering angle, throttle, brake) using
    embeddings and MLPs, then fuses them via a Transformer encoder.

    Args:
        strategic_vocab_size (int): Number of distinct strategic action types.
        tactical_dim (int): Dimensionality of the tactical action vector.
        latent_dim (int): Output dimensionality of the encoded action representation.
        num_heads (int): Number of attention heads in the fusion transformer. Default: 8.
        depth (int): Number of transformer encoder layers. Default: 2.
        dropout (float): Dropout probability. Default: 0.1.
    """

    def __init__(
        self,
        strategic_vocab_size: int = 10,
        tactical_dim: int = 3,
        latent_dim: int = 128,
        num_heads: int = 8,
        depth: int = 2,
        dropout: float = 0.1
    ):
        super().__init__()

        self.strategic_vocab_size = strategic_vocab_size
        self.tactical_dim = tactical_dim
        self.latent_dim = latent_dim

        # Strategic action embedding
        # Maps discrete action IDs to continuous embeddings
        self.strategic_embedding = nn.Embedding(
            num_embeddings=strategic_vocab_size,
            embedding_dim=latent_dim // 2
        )

        # Tactical action MLP
        # Processes continuous action values (e.g., steer, throttle, brake)
        self.tactical_mlp = nn.Sequential(
            nn.Linear(tactical_dim, latent_dim // 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(latent_dim // 4, latent_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # Fusion module
        # Combines strategic and tactical representations using self-attention
        fusion_layer = nn.TransformerEncoderLayer(
            d_model=latent_dim // 2,
            nhead=num_heads,
            dim_feedforward=latent_dim * 2,
            dropout=dropout,
            activation='relu',
            batch_first=True
        )
        self.fusion_transformer = nn.TransformerEncoder(
            fusion_layer,
            num_layers=depth
        )

        # Output projection to latent_dim
        self.output_proj = nn.Linear(latent_dim // 2, latent_dim)

    def forward(self, strategic_action: torch.Tensor, tactical_action: torch.Tensor) -> torch.Tensor:
        """Encode hierarchical actions into a latent representation.

        Args:
            strategic_action (torch.Tensor): Strategic action indices, shape (B,) or (B, 1).
            tactical_action (torch.Tensor): Tactical action values, shape (B, tactical_dim).

        Returns:
            torch.Tensor: Encoded action representation, shape (B, latent_dim).
        """
        # Handle both (B,) and (B, 1) shapes for strategic_action
        if strategic_action.dim() == 2:
            strategic_action = strategic_action.squeeze(-1)

        # Embed strategic action
        strategic_embed = self.strategic_embedding(strategic_action)  # (B, latent_dim//2)

        # Process tactical action through MLP
        tactical_embed = self.tactical_mlp(tactical_action)  # (B, latent_dim//2)

        # Stack for transformer: (B, 2, latent_dim//2)
        # Position 0: strategic, Position 1: tactical
        combined = torch.stack([strategic_embed, tactical_embed], dim=1)

        # Apply fusion transformer
        fused = self.fusion_transformer(combined)  # (B, 2, latent_dim//2)

        # Pool across sequence dimension (mean pooling)
        pooled = fused.mean(dim=1)  # (B, latent_dim//2)

        # Project to final latent dimension
        action_latent = self.output_proj(pooled)  # (B, latent_dim)

        return action_latent
