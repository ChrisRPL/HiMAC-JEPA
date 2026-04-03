import torch

from src.models.himac_jepa import HiMACJEPA
from src.training.targets import build_target_latent


def build_test_model():
    config = {
        "model": {
            "latent_dim": 128,
            "camera_encoder": {"depth": 2, "dropout": 0.0},
            "lidar_encoder": {"dropout": 0.0},
            "radar_encoder": {"dropout": 0.0},
            "action_encoder": {
                "strategic_vocab_size": 10,
                "tactical_dim": 3,
                "latent_dim": 128,
                "num_heads": 8,
                "depth": 1,
                "dropout": 0.0,
            },
        },
        "trajectory_head": {"output_dim": 30},
        "motion_prediction_head": {"output_dim": 60},
        "bev_segmentation_head": {"bev_h": 20, "bev_w": 20, "num_classes": 5},
    }
    return HiMACJEPA(config)


def test_build_target_latent_restores_training_mode():
    model = build_test_model()
    model.train()

    target_latent = build_target_latent(
        model,
        torch.randn(2, 3, 224, 224),
        torch.randn(2, 1024, 3),
        torch.randn(2, 1, 64, 64),
    )

    assert target_latent.shape == (2, 128)
    assert model.training is True


def test_target_latent_is_observation_only():
    model = build_test_model()
    model.eval()

    camera = torch.randn(2, 3, 224, 224)
    lidar = torch.randn(2, 1024, 3)
    radar = torch.randn(2, 1, 64, 64)

    strategic_a = torch.tensor([0, 1])
    tactical_a = torch.zeros(2, 3)
    strategic_b = torch.tensor([7, 8])
    tactical_b = torch.full((2, 3), 10.0)

    with torch.no_grad():
        target_latent = build_target_latent(model, camera, lidar, radar)
        obs_latent = model.encode_observations(camera, lidar, radar)
        mu_a, _, _, _, _ = model(camera, lidar, radar, strategic_a, tactical_a)
        mu_b, _, _, _, _ = model(camera, lidar, radar, strategic_b, tactical_b)

    assert torch.allclose(target_latent, obs_latent, atol=1e-6)
    assert not torch.allclose(mu_a, mu_b, atol=1e-5)
