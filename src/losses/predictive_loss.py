import torch
import torch.nn as nn

class KLDivergenceLoss(nn.Module):
    """
    Calculates the KL-Divergence between two Gaussian distributions.
    Assumes diagonal covariance matrices.

    Args:
        reduction (str): Specifies the reduction to apply to the output: 'none' | 'mean' | 'sum'.
                         'none': no reduction will be applied,
                         'mean': the sum of the output will be divided by the number of elements in the output,
                         'sum': the output will be summed.
    """
    def __init__(self, reduction: str = 'mean'):
        super().__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f"Reduction must be 'none', 'mean', or 'sum', but got {reduction}")
        self.reduction = reduction

    def forward(self, mu_p: torch.Tensor, log_var_p: torch.Tensor, mu_q: torch.Tensor, log_var_q: torch.Tensor) -> torch.Tensor:
        """
        Calculates KL(p || q) where p and q are Gaussian distributions.
        p ~ N(mu_p, exp(log_var_p))
        q ~ N(mu_q, exp(log_var_q))

        KL(p || q) = 0.5 * (log_var_q - log_var_p + (exp(log_var_p) + (mu_p - mu_q).pow(2)) / exp(log_var_q) - 1)
        """
        var_p = torch.exp(log_var_p)
        var_q = torch.exp(log_var_q)

        kl_loss = 0.5 * (log_var_q - log_var_p + (var_p + (mu_p - mu_q).pow(2)) / var_q - 1)

        if self.reduction == 'mean':
            return torch.mean(kl_loss)
        elif self.reduction == 'sum':
            return torch.sum(kl_loss)
        else:
            return kl_loss

class NLLLoss(nn.Module):
    """
    Calculates the Negative Log-Likelihood (NLL) of target data given a predicted Gaussian distribution.
    Assumes diagonal covariance matrices.

    Args:
        reduction (str): Specifies the reduction to apply to the output: 'none' | 'mean' | 'sum'.
                         'none': no reduction will be applied,
                         'mean': the sum of the output will be divided by the number of elements in the output,
                         'sum': the output will be summed.
    """
    def __init__(self, reduction: str = 'mean'):
        super().__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f"Reduction must be 'none', 'mean', or 'sum', but got {reduction}")
        self.reduction = reduction

    def forward(self, mu: torch.Tensor, log_var: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Calculates -log P(target | N(mu, exp(log_var)))
        NLL = 0.5 * (log_var + (target - mu)^2 / exp(log_var))
        """
        nll_loss = 0.5 * (log_var + (target - mu).pow(2) / torch.exp(log_var))

        if self.reduction == 'mean':
            return torch.mean(nll_loss)
        elif self.reduction == 'sum':
            return torch.sum(nll_loss)
        else:
            return nll_loss
