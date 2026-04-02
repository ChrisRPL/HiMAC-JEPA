import torch
import torch.optim as optim
import yaml

from src.losses.predictive_loss import KLDivergenceLoss
from src.losses.vicreg_loss import VICRegLoss
from src.models.himac_jepa import HiMACJEPA


def update_ema_params(model, ema_model, decay):
    with torch.no_grad():
        for ema_p, model_p in zip(ema_model.parameters(), model.parameters()):
            ema_p.mul_(decay).add_(model_p, alpha=1 - decay)


def build_model_config():
    with open("configs/config.yaml", "r") as f:
        root_config = yaml.safe_load(f)
    with open("configs/model/default.yaml", "r") as f:
        model_config = yaml.safe_load(f)
    with open("configs/training/default.yaml", "r") as f:
        training_config = yaml.safe_load(f)

    return {
        "model": model_config,
        "trajectory_head": root_config["trajectory_head"],
        "motion_prediction_head": root_config["motion_prediction_head"],
        "bev_segmentation_head": root_config["bev_segmentation_head"],
        "training": training_config,
    }


def test_training_step():
    cfg = build_model_config()

    model = HiMACJEPA(cfg)
    ema_model = HiMACJEPA(cfg)
    ema_model.load_state_dict(model.state_dict())
    for param in ema_model.parameters():
        param.requires_grad = False

    predictive_loss_fn = KLDivergenceLoss(reduction="mean")
    vicreg_loss_fn = VICRegLoss(
        lambda_param=cfg["training"]["lambda_param"],
        mu_param=cfg["training"]["mu_param"],
        nu_param=cfg["training"]["nu_param"],
    )
    optimizer = optim.Adam(model.parameters(), lr=float(cfg["training"]["learning_rate"]))

    batch_size = 4
    camera = torch.randn(batch_size, 3, 224, 224)
    lidar = torch.randn(batch_size, 1024, 3)
    radar = torch.randn(batch_size, 1, 64, 64)
    strategic_action = torch.randint(
        0, cfg["model"]["action_encoder"]["strategic_vocab_size"], (batch_size,)
    )
    tactical_action = torch.randn(batch_size, cfg["model"]["action_encoder"]["tactical_dim"])

    mu, log_var, _, _, _ = model(camera, lidar, radar, strategic_action, tactical_action)

    with torch.no_grad():
        ema_mu, _, _, _, _ = ema_model(
            camera, lidar, radar, strategic_action, tactical_action
        )
        target_latent = ema_mu.detach()

    predictive_loss = predictive_loss_fn(
        mu, log_var, target_latent, torch.zeros_like(log_var)
    )
    vicreg_loss = vicreg_loss_fn(mu, target_latent)
    total_loss = predictive_loss + vicreg_loss

    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()

    update_ema_params(model, ema_model, decay=cfg["training"]["ema_decay"])

    assert total_loss.item() > 0, "Total loss should be positive"
    assert not torch.isnan(total_loss).any(), "Total loss should not be NaN"
    assert not torch.isinf(total_loss).any(), "Total loss should not be Inf"
