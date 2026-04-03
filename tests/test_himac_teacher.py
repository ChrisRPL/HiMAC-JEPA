import torch

from src.models.himac_jepa import HiMACJEPA
from src.models.himac_teacher import HiMACObservationTeacher


def build_model_config():
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


def test_teacher_matches_student_observation_path():
    config = build_model_config()
    student = HiMACJEPA(config)
    teacher = HiMACObservationTeacher(config)
    teacher.load_from_student(student)

    camera = torch.randn(2, 3, 224, 224)
    lidar = torch.randn(2, 1024, 3)
    radar = torch.randn(2, 1, 64, 64)

    with torch.no_grad():
        student_latent = student.encode_observations(camera, lidar, radar)
        teacher_latent = teacher(camera, lidar, radar)

    assert torch.allclose(student_latent, teacher_latent, atol=1e-6)


def test_teacher_ema_updates_observation_weights():
    config = build_model_config()
    student = HiMACJEPA(config)
    teacher = HiMACObservationTeacher(config)
    teacher.load_from_student(student)

    initial = teacher.camera_encoder.proj.weight.detach().clone()
    student.camera_encoder.proj.weight.data.add_(1.0)

    teacher.update_from_student(student, decay=0.5)

    assert not torch.equal(initial, teacher.camera_encoder.proj.weight)


def test_teacher_copies_batch_norm_buffers_from_student():
    config = build_model_config()
    student = HiMACJEPA(config)
    teacher = HiMACObservationTeacher(config)
    teacher.load_from_student(student)

    student.train()
    camera = torch.randn(4, 3, 224, 224)
    lidar = torch.randn(4, 1024, 3)
    radar = torch.randn(4, 1, 64, 64)
    strategic = torch.randint(0, 10, (4,))
    tactical = torch.randn(4, 3)

    student(camera, lidar, radar, strategic, tactical)
    teacher.update_from_student(student, decay=0.9)

    assert torch.allclose(
        teacher.lidar_encoder.mlp1[1].running_mean,
        student.lidar_encoder.mlp1[1].running_mean,
    )
