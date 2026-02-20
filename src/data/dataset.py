import torch
from torch.utils.data import Dataset, DataLoader

class MultiModalDrivingDataset(Dataset):
    """Skeleton for multi-modal driving dataset (e.g., nuScenes, Waymo)."""
    def __init__(self, config, split='train'):
        self.config = config
        self.split = split
        # In a real implementation, load metadata and file paths here
        self.samples = range(100) # Dummy samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        # Dummy data generation matching expected shapes
        camera = torch.randn(3, 224, 224)
        lidar = torch.randn(1024, 3)
        radar = torch.randn(1, 64, 64)
        
        # Hierarchical actions: [Strategic, Tactical]
        # Strategic: e.g., 0: Keep Lane, 1: Change Left, 2: Change Right
        # Tactical: e.g., [Steering, Acceleration, Braking]
        strategic_action = torch.tensor(0)
        tactical_action = torch.randn(3)
        
        return {
            'camera': camera,
            'lidar': lidar,
            'radar': radar,
            'strategic_action': strategic_action,
            'tactical_action': tactical_action
        }

def get_dataloader(config, split='train'):
    dataset = MultiModalDrivingDataset(config, split)
    return DataLoader(
        dataset,
        batch_size=config['data']['batch_size'],
        shuffle=(split == 'train'),
        num_workers=config['data']['num_workers']
    )
