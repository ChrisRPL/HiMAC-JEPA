import torch
from torch.utils.data import Dataset, DataLoader
import os
import numpy as np
from PIL import Image

class MultiModalDrivingDataset(Dataset):
    """Multi-modal driving dataset for nuScenes/Waymo-like data.

    This class provides a flexible data pipeline for loading multi-modal sensor data
    (camera, LiDAR, radar) along with associated actions. It supports metadata handling,
    file path management, and graceful handling of missing data files.
    Data augmentation and custom collate functions are also integrated.
    """
    def __init__(self, config, split='train', dataset_root='/tmp/dummy_dataset'):
        self.config = config
        self.split = split
        self.dataset_root = dataset_root
        self.data_augmentation = config['data'].get('augmentation', False)

        # In a real implementation, metadata would be loaded from JSON/pickle files
        # For this simulation, we create dummy metadata.
        self.metadata = self._generate_dummy_metadata(num_samples=config['data'].get('num_samples', 100))

        if not self.metadata:
            raise RuntimeError("No samples found in metadata. Ensure dataset_root is correct and metadata is properly loaded.")

    def _generate_dummy_metadata(self, num_samples):
        """Generates dummy metadata mimicking nuScenes/Waymo structure.

        In a real scenario, this would parse dataset manifests (e.g., JSON files)
        to create a list of sample dictionaries, each containing paths to sensor data
        and associated annotations.
        """
        dummy_metadata = []
        for i in range(num_samples):
            sample_id = f"sample_{i:06d}"
            # Simulate file paths. These files won't actually exist in the sandbox.
            dummy_metadata.append({
                'sample_id': sample_id,
                'camera_path': os.path.join(self.dataset_root, 'samples', 'CAM_FRONT', f'{sample_id}.jpg'),
                'lidar_path': os.path.join(self.dataset_root, 'samples', 'LIDAR_TOP', f'{sample_id}.bin'),
                'radar_path': os.path.join(self.dataset_root, 'samples', 'RADAR_FRONT', f'{sample_id}.pcd'),
                'strategic_action': i % 3, # Dummy strategic action
                'tactical_action': np.random.rand(3).tolist() # Dummy tactical action
            })
        return dummy_metadata

    def __len__(self):
        return len(self.metadata)

    def _load_sensor_data(self, path, sensor_type):
        """Simulates loading sensor data from a given path.

        In a real implementation, this would load actual image, point cloud, or radar data.
        Here, it generates dummy data but includes file existence checks.
        """
        if not os.path.exists(path):
            # For demonstration, we generate dummy data if file doesn't exist
            # In a real scenario, this might raise an error or return None/empty data
            # depending on how missing data is handled.
            print(f"Warning: {sensor_type} file not found at {path}. Generating dummy data.")
            if sensor_type == 'camera':
                return torch.randn(3, 224, 224) # Dummy camera image (C, H, W)
            elif sensor_type == 'lidar':
                return torch.randn(self.config['data']['lidar_points'], 3) # Dummy LiDAR points (N, 3)
            elif sensor_type == 'radar':
                return torch.randn(1, 64, 64) # Dummy radar data (C, H, W)
            else:
                raise ValueError(f"Unknown sensor type: {sensor_type}")
        
        # Simulate loading data if file existed (which it won't in this sandbox)
        if sensor_type == 'camera':
            # Example: Image.open(path).convert('RGB').resize(...)
            return torch.randn(3, 224, 224)
        elif sensor_type == 'lidar':
            # Example: np.fromfile(path, dtype=np.float32).reshape(-1, 4)[:, :3]
            return torch.randn(self.config['data']['lidar_points'], 3)
        elif sensor_type == 'radar':
            # Example: np.fromfile(path, dtype=np.float32).reshape(...)
            return torch.randn(1, 64, 64)
        else:
            raise ValueError(f"Unknown sensor type: {sensor_type}")

    def _apply_augmentation(self, camera, lidar, radar):
        """Applies data augmentation to the sensor data.

        This is a placeholder for actual augmentation techniques.
        """
        if self.data_augmentation:
            # Example: RandomCrop, ColorJitter for camera; random rotation for lidar/radar
            camera = camera + torch.randn_like(camera) * 0.01
            lidar = lidar + torch.randn_like(lidar) * 0.01
            radar = radar + torch.randn_like(radar) * 0.01
        return camera, lidar, radar

    def __getitem__(self, idx):
        sample_info = self.metadata[idx]

        camera = self._load_sensor_data(sample_info['camera_path'], 'camera')
        lidar = self._load_sensor_data(sample_info['lidar_path'], 'lidar')
        radar = self._load_sensor_data(sample_info['radar_path'], 'radar')

        camera, lidar, radar = self._apply_augmentation(camera, lidar, radar)

        strategic_action = torch.tensor(sample_info['strategic_action'], dtype=torch.long)
        tactical_action = torch.tensor(sample_info['tactical_action'], dtype=torch.float32)
        
        return {
            'camera': camera,
            'lidar': lidar,
            'radar': radar,
            'strategic_action': strategic_action,
            'tactical_action': tactical_action,
            'sample_id': sample_info['sample_id']
        }

def collate_fn(batch):
    """Custom collate function for handling variable-sized data (e.g., LiDAR point clouds).

    This function pads LiDAR data to the maximum number of points in the batch
    and stacks other tensors. It's essential for creating batches when sensor
    data might have varying dimensions.
    """
    # Separate out the different modalities and metadata
    cameras = torch.stack([item['camera'] for item in batch])
    radars = torch.stack([item['radar'] for item in batch])
    strategic_actions = torch.stack([item['strategic_action'] for item in batch])
    tactical_actions = torch.stack([item['tactical_action'] for item in batch])
    sample_ids = [item['sample_id'] for item in batch]

    # Handle LiDAR: pad to max points in batch
    lidar_points = [item['lidar'] for item in batch]
    max_points = max(l.shape[0] for l in lidar_points)
    padded_lidars = []
    for l in lidar_points:
        # Pad with zeros
        padding = torch.zeros(max_points - l.shape[0], l.shape[1], dtype=l.dtype)
        padded_lidars.append(torch.cat([l, padding], dim=0))
    lidars = torch.stack(padded_lidars)

    return {
        'camera': cameras,
        'lidar': lidars,
        'radar': radars,
        'strategic_action': strategic_actions,
        'tactical_action': tactical_actions,
        'sample_id': sample_ids
    }

def get_dataloader(config, split='train'):
    """Returns a DataLoader for the MultiModalDrivingDataset.

    Args:
        config (dict): Configuration dictionary containing dataset parameters.
        split (str): Dataset split, e.g., 'train', 'val', 'test'.

    Returns:
        torch.utils.data.DataLoader: Configured DataLoader instance.
    """
    dataset = MultiModalDrivingDataset(config, split)
    return DataLoader(
        dataset,
        batch_size=config['data']['batch_size'],
        shuffle=(split == 'train'),
        num_workers=config['data']['num_workers'],
        collate_fn=collate_fn # Use custom collate function
    )
