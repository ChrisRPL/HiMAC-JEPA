from pathlib import Path

import torch

from src.training.checkpointing import save_training_checkpoints


def test_save_training_checkpoints_writes_latest_and_best(tmp_path):
    model = torch.nn.Linear(4, 2)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    paths = save_training_checkpoints(
        model=model,
        optimizer=optimizer,
        checkpoint_root=tmp_path,
        experiment_name="demo-exp",
        epoch=3,
        avg_total_loss=1.25,
        config={"experiment_name": "demo-exp"},
        best_loss=None,
    )

    assert paths["saved_best"] is True
    assert paths["latest_path"].exists()
    assert paths["best_path"].exists()
    assert paths["latest_alias_path"].exists()
    assert paths["best_alias_path"].exists()

    checkpoint = torch.load(paths["best_path"], map_location="cpu")
    assert checkpoint["epoch"] == 3
    assert checkpoint["avg_total_loss"] == 1.25
    assert checkpoint["config"]["experiment_name"] == "demo-exp"


def test_save_training_checkpoints_preserves_best_on_regression(tmp_path):
    model = torch.nn.Linear(4, 2)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    first_paths = save_training_checkpoints(
        model=model,
        optimizer=optimizer,
        checkpoint_root=tmp_path,
        experiment_name="demo-exp",
        epoch=1,
        avg_total_loss=0.5,
        config={"experiment_name": "demo-exp"},
        best_loss=None,
    )
    initial_best = torch.load(first_paths["best_path"], map_location="cpu")

    second_paths = save_training_checkpoints(
        model=model,
        optimizer=optimizer,
        checkpoint_root=tmp_path,
        experiment_name="demo-exp",
        epoch=2,
        avg_total_loss=0.8,
        config={"experiment_name": "demo-exp"},
        best_loss=0.5,
    )

    assert second_paths["saved_best"] is False
    retained_best = torch.load(first_paths["best_path"], map_location="cpu")
    latest_checkpoint = torch.load(second_paths["latest_path"], map_location="cpu")

    assert retained_best["epoch"] == initial_best["epoch"] == 1
    assert latest_checkpoint["epoch"] == 2
