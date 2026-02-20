import torch
import pytest

from src.losses.predictive_loss import KLDivergenceLoss, NLLLoss
from src.losses.vicreg_loss import VICRegLoss

# --- KLDivergenceLoss Tests ---

def test_kl_divergence_loss_basic():
    mu_p = torch.tensor([0.0])
    log_var_p = torch.tensor([0.0])  # var_p = 1.0
    mu_q = torch.tensor([0.0])
    log_var_q = torch.tensor([0.0])  # var_q = 1.0
    loss_fn = KLDivergenceLoss(reduction=\'sum\')
    loss = loss_fn(mu_p, log_var_p, mu_q, log_var_q)
    assert torch.isclose(loss, torch.tensor(0.0)), f"Expected 0.0, got {loss.item()}"

def test_kl_divergence_loss_shifted_mean():
    mu_p = torch.tensor([1.0])
    log_var_p = torch.tensor([0.0])
    mu_q = torch.tensor([0.0])
    log_var_q = torch.tensor([0.0])
    loss_fn = KLDivergenceLoss(reduction=\'sum\')
    loss = loss_fn(mu_p, log_var_p, mu_q, log_var_q)
    # KL(N(1,1) || N(0,1)) = 0.5 * (log(1/1) + (1 + (1-0)^2)/1 - 1) = 0.5 * (0 + (1+1)/1 - 1) = 0.5 * (2-1) = 0.5
    assert torch.isclose(loss, torch.tensor(0.5)), f"Expected 0.5, got {loss.item()}"

def test_kl_divergence_loss_different_variance():
    mu_p = torch.tensor([0.0])
    log_var_p = torch.tensor([0.0])  # var_p = 1.0
    mu_q = torch.tensor([0.0])
    log_var_q = torch.tensor([torch.log(torch.tensor(2.0))]) # var_q = 2.0
    loss_fn = KLDivergenceLoss(reduction=\'sum\')
    loss = loss_fn(mu_p, log_var_p, mu_q, log_var_q)
    # KL(N(0,1) || N(0,2)) = 0.5 * (log(2/1) + (1 + 0)/2 - 1) = 0.5 * (0.6931 + 0.5 - 1) = 0.5 * 0.1931 = 0.09655
    expected_loss = 0.5 * (torch.log(torch.tensor(2.0)) + 0.5 - 1)
    assert torch.isclose(loss, expected_loss), f"Expected {expected_loss.item()}, got {loss.item()}"

def test_kl_divergence_loss_batch():
    mu_p = torch.tensor([[0.0, 0.0], [1.0, 1.0]])
    log_var_p = torch.tensor([[0.0, 0.0], [0.0, 0.0]])
    mu_q = torch.tensor([[0.0, 0.0], [0.0, 0.0]])
    log_var_q = torch.tensor([[0.0, 0.0], [0.0, 0.0]])
    loss_fn = KLDivergenceLoss(reduction=\'mean\')
    loss = loss_fn(mu_p, log_var_p, mu_q, log_var_q)
    assert torch.isclose(loss, torch.tensor(0.0)), f"Expected 0.0, got {loss.item()}"

    mu_p = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    log_var_p = torch.tensor([[0.0, 0.0], [0.0, 0.0]])
    mu_q = torch.tensor([[0.0, 0.0], [0.0, 0.0]])
    log_var_q = torch.tensor([[0.0, 0.0], [0.0, 0.0]])
    loss_fn = KLDivergenceLoss(reduction=\'mean\')
    loss = loss_fn(mu_p, log_var_p, mu_q, log_var_q)
    # Each element contributes 0.5, total 4 elements, mean = (0.5 * 4) / 4 = 0.5
    assert torch.isclose(loss, torch.tensor(0.5)), f"Expected 0.5, got {loss.item()}"

# --- NLLLoss Tests ---

def test_nll_loss_basic():
    mu = torch.tensor([0.0])
    log_var = torch.tensor([0.0]) # var = 1.0
    target = torch.tensor([0.0])
    loss_fn = NLLLoss(reduction=\'sum\')
    loss = loss_fn(mu, log_var, target)
    # NLL = 0.5 * (log_var + (target - mu)^2 / exp(log_var)) = 0.5 * (0 + (0-0)^2 / 1) = 0.0
    assert torch.isclose(loss, torch.tensor(0.0)), f"Expected 0.0, got {loss.item()}"

def test_nll_loss_shifted_target():
    mu = torch.tensor([0.0])
    log_var = torch.tensor([0.0]) # var = 1.0
    target = torch.tensor([1.0])
    loss_fn = NLLLoss(reduction=\'sum\')
    loss = loss_fn(mu, log_var, target)
    # NLL = 0.5 * (0 + (1-0)^2 / 1) = 0.5
    assert torch.isclose(loss, torch.tensor(0.5)), f"Expected 0.5, got {loss.item()}"

def test_nll_loss_different_variance():
    mu = torch.tensor([0.0])
    log_var = torch.tensor([torch.log(torch.tensor(2.0))]) # var = 2.0
    target = torch.tensor([0.0])
    loss_fn = NLLLoss(reduction=\'sum\')
    loss = loss_fn(mu, log_var, target)
    # NLL = 0.5 * (log(2) + (0-0)^2 / 2) = 0.5 * log(2) = 0.34657
    expected_loss = 0.5 * torch.log(torch.tensor(2.0))
    assert torch.isclose(loss, expected_loss), f"Expected {expected_loss.item()}, got {loss.item()}"

def test_nll_loss_batch():
    mu = torch.tensor([[0.0, 0.0], [1.0, 1.0]])
    log_var = torch.tensor([[0.0, 0.0], [0.0, 0.0]])
    target = torch.tensor([[0.0, 0.0], [1.0, 1.0]])
    loss_fn = NLLLoss(reduction=\'mean\')
    loss = loss_fn(mu, log_var, target)
    assert torch.isclose(loss, torch.tensor(0.0)), f"Expected 0.0, got {loss.item()}"

    mu = torch.tensor([[0.0, 0.0], [0.0, 0.0]])
    log_var = torch.tensor([[0.0, 0.0], [0.0, 0.0]])
    target = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    loss_fn = NLLLoss(reduction=\'mean\')
    loss = loss_fn(mu, log_var, target)
    # Each element contributes 0.5, total 4 elements, mean = (0.5 * 4) / 4 = 0.5
    assert torch.isclose(loss, torch.tensor(0.5)), f"Expected 0.5, got {loss.item()}"

# --- VICRegLoss Tests ---

def test_vicreg_loss_basic():
    z_a = torch.randn(16, 128)
    z_b = torch.randn(16, 128)
    loss_fn = VICRegLoss()
    loss = loss_fn(z_a, z_b)
    assert loss.item() > 0, "VICReg loss should be positive for random inputs"

def test_vicreg_loss_identical_inputs():
    z_a = torch.randn(16, 128)
    z_b = z_a.clone()
    loss_fn = VICRegLoss(lambda_param=1.0, mu_param=0.0, nu_param=0.0) # Only invariance term
    loss = loss_fn(z_a, z_b)
    assert torch.isclose(loss, torch.tensor(0.0)), "Invariance term should be zero for identical inputs"

def test_vicreg_loss_variance_term():
    z_a = torch.ones(16, 128) # All ones, variance is 0
    z_b = torch.ones(16, 128)
    loss_fn = VICRegLoss(lambda_param=0.0, mu_param=1.0, nu_param=0.0) # Only variance term
    loss = loss_fn(z_a, z_b)
    # std_z_a will be 0, 1 - std_z_a will be 1, ReLU(1) = 1. So std_loss = 1 + 1 = 2
    assert torch.isclose(loss, torch.tensor(2.0)), "Variance term should be 2.0 for all-ones inputs"

def test_vicreg_loss_covariance_term():
    # Create inputs where off-diagonal covariance is non-zero
    z_a = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32)
    z_b = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32)
    loss_fn = VICRegLoss(lambda_param=0.0, mu_param=0.0, nu_param=1.0, eps=1e-5) # Only covariance term
    loss = loss_fn(z_a, z_b)
    # For a 2x2 matrix, off-diagonal elements are (0,1) and (1,0). If centered, cov will have non-zero off-diagonals.
    # The exact value is complex to calculate by hand, but it should be positive.
    assert loss.item() > 0, "Covariance term should be positive for inputs with non-zero off-diagonal covariance"

def test_vicreg_loss_off_diagonal():
    loss_fn = VICRegLoss()
    x = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=torch.float32)
    off_diag = loss_fn.off_diagonal(x)
    expected_off_diag = torch.tensor([2., 3., 4., 6., 7., 8.])
    assert torch.allclose(off_diag, expected_off_diag), f"Expected {expected_off_diag}, got {off_diag}"

# Note: These tests require PyTorch to be installed to run. If PyTorch is not installed,
# these tests will fail with a ModuleNotFoundError. This is expected given the current
# environment constraints.
