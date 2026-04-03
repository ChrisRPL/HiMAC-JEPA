import torch
import torch.optim as optim
import yaml

from src.losses.predictive_loss import KLDivergenceLoss
from src.losses.vicreg_loss import VICRegLoss
from src.masking.spatio_temporal_masking import SpatioTemporalMasking
from src.models.himac_jepa import HiMACJEPA
from src.training.masking import build_batch_masks
from src.training.targets import build_ema_teacher, build_target_latent, update_ema_teacher


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


def build_temporal_test_config():
    cfg = build_model_config()
    cfg["model"]["camera_encoder"]["depth"] = 2
    cfg["model"]["camera_encoder"]["dropout"] = 0.0
    cfg["model"]["lidar_encoder"]["dropout"] = 0.0
    cfg["model"]["radar_encoder"]["dropout"] = 0.0
    cfg["model"]["action_encoder"]["depth"] = 1
    cfg["model"]["action_encoder"]["dropout"] = 0.0
    return cfg


def test_training_step():
    cfg = build_model_config()

    model = HiMACJEPA(cfg)
    ema_teacher = build_ema_teacher(cfg, student=model)

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

    target_latent = build_target_latent(ema_teacher, camera, lidar, radar)

    predictive_loss = predictive_loss_fn(
        mu, log_var, target_latent, torch.zeros_like(log_var)
    )
    vicreg_loss = vicreg_loss_fn(mu, target_latent)
    total_loss = predictive_loss + vicreg_loss

    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()

    update_ema_teacher(model, ema_teacher, decay=cfg["training"]["ema_decay"])

    assert total_loss.item() > 0, "Total loss should be positive"
    assert not torch.isnan(total_loss).any(), "Total loss should not be NaN"
    assert not torch.isinf(total_loss).any(), "Total loss should not be Inf"


def test_temporal_training_step_with_masking():
    cfg = build_temporal_test_config()

    model = HiMACJEPA(cfg)
    ema_teacher = build_ema_teacher(cfg, student=model)

    predictive_loss_fn = KLDivergenceLoss(reduction="mean")
    vicreg_loss_fn = VICRegLoss(
        lambda_param=cfg["training"]["lambda_param"],
        mu_param=cfg["training"]["mu_param"],
        nu_param=cfg["training"]["nu_param"],
    )
    optimizer = optim.Adam(model.parameters(), lr=float(cfg["training"]["learning_rate"]))
    masker = SpatioTemporalMasking(mask_ratio_spatial=0.5, mask_ratio_temporal=0.5)

    batch_size = 2
    context_steps = 4
    target_steps = 2
    vocab_size = cfg["model"]["action_encoder"]["strategic_vocab_size"]
    tactical_dim = cfg["model"]["action_encoder"]["tactical_dim"]

    context_camera = torch.randn(batch_size, context_steps, 3, 224, 224)
    context_lidar = torch.randn(batch_size, context_steps, 1024, 3)
    context_radar = torch.randn(batch_size, context_steps, 1, 64, 64)
    context_strategic = torch.randint(0, vocab_size, (batch_size, context_steps))
    context_tactical = torch.randn(batch_size, context_steps, tactical_dim)

    target_camera = torch.randn(batch_size, target_steps, 3, 224, 224)
    target_lidar = torch.randn(batch_size, target_steps, 1024, 3)
    target_radar = torch.randn(batch_size, target_steps, 1, 64, 64)
    target_strategic = torch.randint(0, vocab_size, (batch_size, target_steps))
    target_tactical = torch.randn(batch_size, target_steps, tactical_dim)

    masks = build_batch_masks(masker, context_camera, context_lidar, context_radar)

    mu, log_var, _, _, _ = model(
        context_camera,
        context_lidar,
        context_radar,
        context_strategic,
        context_tactical,
        masks,
    )

    target_latent = build_target_latent(
        ema_teacher,
        target_camera,
        target_lidar,
        target_radar,
    )

    predictive_loss = predictive_loss_fn(
        mu, log_var, target_latent, torch.zeros_like(log_var)
    )
    vicreg_loss = vicreg_loss_fn(mu, target_latent)
    total_loss = predictive_loss + vicreg_loss

    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()

    update_ema_teacher(model, ema_teacher, decay=cfg["training"]["ema_decay"])

    assert masks["temporal"].shape == (batch_size, context_steps)
    assert total_loss.item() > 0, "Temporal total loss should be positive"
    assert not torch.isnan(total_loss).any(), "Temporal total loss should not be NaN"
    assert not torch.isinf(total_loss).any(), "Temporal total loss should not be Inf"
