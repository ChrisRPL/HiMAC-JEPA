"""Temporal fusion module for aggregating sequence features."""
import torch
import torch.nn as nn
import math


class TemporalTransformer(nn.Module):
    """Transformer-based temporal aggregation for multi-frame features."""

    def __init__(self, d_model=768, nhead=8, num_layers=4, dropout=0.1):
        """
        Initialize temporal transformer.

        Args:
            d_model: Feature dimension
            nhead: Number of attention heads
            num_layers: Number of transformer layers
            dropout: Dropout probability
        """
        super().__init__()

        self.d_model = d_model

        # Positional encoding for temporal position
        self.pos_encoder = PositionalEncoding(d_model, dropout)

        # Transformer encoder for temporal aggregation
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=False  # Expects (T, B, D)
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

        # Layer norm for output
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, aggregate='last') -> torch.Tensor:
        """
        Aggregate temporal features.

        Args:
            x: Input features of shape (B, T, D)
            aggregate: Aggregation method ('last', 'mean', 'max', 'all')

        Returns:
            Aggregated features of shape (B, D) or (B, T, D) if aggregate='all'
        """
        B, T, D = x.shape

        # Transpose to (T, B, D) for transformer
        x = x.transpose(0, 1)

        # Add positional encoding
        x = self.pos_encoder(x)

        # Apply transformer
        x = self.transformer(x)  # (T, B, D)

        # Apply final norm
        x = self.norm(x)

        # Aggregate based on method
        if aggregate == 'last':
            # Use final timestep
            out = x[-1]  # (B, D)
        elif aggregate == 'mean':
            # Average over time
            out = x.mean(dim=0)  # (B, D)
        elif aggregate == 'max':
            # Max pool over time
            out = x.max(dim=0)[0]  # (B, D)
        elif aggregate == 'all':
            # Return all timesteps
            out = x.transpose(0, 1)  # (B, T, D)
        else:
            raise ValueError(f"Unknown aggregate method: {aggregate}")

        return out


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for temporal sequences."""

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 100):
        """
        Initialize positional encoding.

        Args:
            d_model: Feature dimension
            dropout: Dropout probability
            max_len: Maximum sequence length
        """
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Create positional encoding matrix
        position = torch.arange(max_len).unsqueeze(1)  # (max_len, 1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )

        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # Register as buffer (not a parameter, but part of state)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add positional encoding to input.

        Args:
            x: Input tensor of shape (T, B, D)

        Returns:
            x with positional encoding added, shape (T, B, D)
        """
        # x shape: (T, B, D)
        # pe shape: (T, D) -> unsqueeze to (T, 1, D) for broadcasting
        x = x + self.pe[:x.size(0), :].unsqueeze(1)
        return self.dropout(x)


class SimpleTem

poralAggregator(nn.Module):
    """Simple temporal aggregation without transformer (faster baseline)."""

    def __init__(self, d_model=768, hidden_dim=None):
        """
        Initialize simple aggregator.

        Args:
            d_model: Feature dimension
            hidden_dim: Hidden dimension for MLP (default: d_model)
        """
        super().__init__()

        if hidden_dim is None:
            hidden_dim = d_model

        # Simple MLP aggregator
        self.mlp = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, d_model)
        )

        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, aggregate='last') -> torch.Tensor:
        """
        Aggregate temporal features with simple MLP.

        Args:
            x: Input features of shape (B, T, D)
            aggregate: Aggregation method ('last', 'mean', 'max')

        Returns:
            Aggregated features of shape (B, D)
        """
        B, T, D = x.shape

        # Aggregate over time first
        if aggregate == 'last':
            x_agg = x[:, -1, :]  # (B, D)
        elif aggregate == 'mean':
            x_agg = x.mean(dim=1)  # (B, D)
        elif aggregate == 'max':
            x_agg = x.max(dim=1)[0]  # (B, D)
        else:
            raise ValueError(f"Unknown aggregate method: {aggregate}")

        # Apply MLP
        out = self.mlp(x_agg)
        out = self.norm(out + x_agg)  # Residual connection

        return out
