import torch
import torch.nn as nn

class VICRegLoss(nn.Module):
    """
    VICReg (Variance-Invariance-Covariance Regularization) Loss.
    Based on the paper: "VICReg: Variance-Invariance-Covariance Regularization for Self-Supervised Learning" by Bardes et al.

    Args:
        lambda_param (float): Weight for the invariance term.
        mu_param (float): Weight for the variance term.
        nu_param (float): Weight for the covariance term.
        eps (float): Epsilon for numerical stability in variance and covariance terms.
    """
    def __init__(self, lambda_param: float = 25.0, mu_param: float = 25.0, nu_param: float = 1.0, eps: float = 1e-5):
        super().__init__()
        self.lambda_param = lambda_param
        self.mu_param = mu_param
        self.nu_param = nu_param
        self.eps = eps

    def forward(self, z_a: torch.Tensor, z_b: torch.Tensor) -> torch.Tensor:
        """
        Calculates the VICReg loss for two sets of embeddings, z_a and z_b.

        Args:
            z_a (torch.Tensor): Embeddings from the first view/branch. Shape: (batch_size, embedding_dim)
            z_b (torch.Tensor): Embeddings from the second view/branch. Shape: (batch_size, embedding_dim)

        Returns:
            torch.Tensor: The total VICReg loss.
        """
        # Ensure inputs are centered
        z_a = z_a - z_a.mean(dim=0)
        z_b = z_b - z_b.mean(dim=0)

        # Invariance term (MSE between z_a and z_b)
        sim_loss = nn.MSELoss(reduction=\'mean\')(z_a, z_b)

        # Variance term
        std_z_a = torch.sqrt(z_a.var(dim=0) + self.eps)
        std_z_b = torch.sqrt(z_b.var(dim=0) + self.eps)
        std_loss = torch.mean(nn.ReLU()(1 - std_z_a)) + torch.mean(nn.ReLU()(1 - std_z_b))

        # Covariance term
        N, D = z_a.shape

        cov_z_a = (z_a.T @ z_a) / (N - 1)
        cov_z_b = (z_b.T @ z_b) / (N - 1)

        cov_loss = self.off_diagonal(cov_z_a).pow_(2).sum().div(D) + self.off_diagonal(cov_z_b).pow_(2).sum().div(D)

        loss = self.lambda_param * sim_loss + self.mu_param * std_loss + self.nu_param * cov_loss
        return loss

    def off_diagonal(self, x: torch.Tensor) -> torch.Tensor:
        """
        Returns a flattened view of the off-diagonal elements of a square matrix.
        """
        n, m = x.shape
        assert n == m
        return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()
