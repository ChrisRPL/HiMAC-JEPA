import torch

from src.models.himac_jepa import HiMACJEPA
from src.training.targets import build_ema_teacher, build_target_latent


def build_test_config():
    return {
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


def test_build_target_latent_restores_teacher_training_mode():
    config = build_test_config()
    student = HiMACJEPA(config)
    teacher = build_ema_teacher(config, student=student)
    teacher.train()

    target_latent = build_target_latent(
        teacher,
        torch.randn(2, 3, 224, 224),
        torch.randn(2, 1024, 3),
        torch.randn(2, 1, 64, 64),
    )

    assert target_latent.shape == (2, 128)
    assert teacher.training is True


def test_build_target_latent_preserves_teacher_eval_mode():
    config = build_test_config()
    student = HiMACJEPA(config)
    teacher = build_ema_teacher(config, student=student)
    teacher.eval()

    _ = build_target_latent(
        teacher,
        torch.randn(2, 3, 224, 224),
        torch.randn(2, 1024, 3),
        torch.randn(2, 1, 64, 64),
    )

    assert teacher.training is False


def test_target_latent_is_observation_only():
    config = build_test_config()
    student = HiMACJEPA(config)
    teacher = build_ema_teacher(config, student=student)
    student.eval()
    teacher.eval()

    camera = torch.randn(2, 3, 224, 224)
    lidar = torch.randn(2, 1024, 3)
    radar = torch.randn(2, 1, 64, 64)

    strategic_a = torch.tensor([0, 1])
    tactical_a = torch.zeros(2, 3)
    strategic_b = torch.tensor([7, 8])
    tactical_b = torch.full((2, 3), 10.0)

    with torch.no_grad():
        target_latent = build_target_latent(teacher, camera, lidar, radar)
        obs_latent = teacher(camera, lidar, radar)
        mu_a, _, _, _, _ = student(camera, lidar, radar, strategic_a, tactical_a)
        mu_b, _, _, _, _ = student(camera, lidar, radar, strategic_b, tactical_b)

    assert torch.allclose(target_latent, obs_latent, atol=1e-6)
    assert not torch.allclose(mu_a, mu_b, atol=1e-5)


def test_temporal_target_latent_stays_observation_only_with_future_actions():
    config = build_test_config()
    student = HiMACJEPA(config)
    teacher = build_ema_teacher(config, student=student)
    student.eval()
    teacher.eval()

    batch_size = 2
    context_steps = 3
    future_steps = 2

    context_camera = torch.randn(batch_size, context_steps, 3, 224, 224)
    context_lidar = torch.randn(batch_size, context_steps, 1024, 3)
    context_radar = torch.randn(batch_size, context_steps, 1, 64, 64)
    context_strategic = torch.randint(0, 10, (batch_size, context_steps))
    context_tactical = torch.randn(batch_size, context_steps, 3)

    target_camera = torch.randn(batch_size, future_steps, 3, 224, 224)
    target_lidar = torch.randn(batch_size, future_steps, 1024, 3)
    target_radar = torch.randn(batch_size, future_steps, 1, 64, 64)

    future_strategic_a = torch.tensor([[0, 1], [2, 3]])
    future_tactical_a = torch.zeros(batch_size, future_steps, 3)
    future_strategic_b = torch.tensor([[4, 5], [6, 7]])
    future_tactical_b = torch.full((batch_size, future_steps, 3), 5.0)

    with torch.no_grad():
        target_latent = build_target_latent(teacher, target_camera, target_lidar, target_radar)
        obs_latent = teacher(target_camera, target_lidar, target_radar)
        mu_a, _, _, _, _ = student(
            context_camera,
            context_lidar,
            context_radar,
            context_strategic,
            context_tactical,
            future_strategic_action=future_strategic_a,
            future_tactical_action=future_tactical_a,
        )
        mu_b, _, _, _, _ = student(
            context_camera,
            context_lidar,
            context_radar,
            context_strategic,
            context_tactical,
            future_strategic_action=future_strategic_b,
            future_tactical_action=future_tactical_b,
        )

    assert torch.allclose(target_latent, obs_latent, atol=1e-6)
    assert not torch.allclose(mu_a, mu_b, atol=1e-5)
