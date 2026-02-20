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

    Args:
        config (dict): Configuration dictionary containing dataset parameters.
        split (str): The dataset split to use, e.g., 'train', 'val', 'test'. Defaults to 'train'.
        dataset_root (str): The root directory where the dataset is expected to be found.
                            Defaults to '/tmp/dummy_dataset' for demonstration purposes.
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

        Args:
            num_samples (int): The number of dummy samples to generate.

        Returns:
            list: A list of dictionaries, where each dictionary represents a sample
                  and contains paths to sensor data and action labels.
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
        """Returns the total number of samples in the dataset.

        Returns:
            int: The number of samples.
        """
        return len(self.metadata)

    def _load_sensor_data(self, path, sensor_type):
        """Simulates loading sensor data from a given path.

        In a real implementation, this would load actual image, point cloud, or radar data.
        Here, it generates dummy data but includes file existence checks.

        Args:
            path (str): The expected file path for the sensor data.
            sensor_type (str): The type of sensor data (e.g., 'camera', 'lidar', 'radar').

        Returns:
            torch.Tensor: A tensor representing the loaded (or dummy generated) sensor data.

        Raises:
            ValueError: If an unknown sensor type is provided.
        """
        if not os.path.exists(path):
            # For demonstration, we generate dummy data if file doesn't exist
            # In a real scenario, this might raise an error or return None/empty data
            # depending on how missing data is handled.
            print(f"Warning: {sensor_type} file not found at {path}. Generating dummy data.")
            if sensor_type == 'camera':
                return torch.randn(3, 224, 224) # Dummy camera image (C, H, W)
            elif sensor_type == 'lidar':
                # Ensure 'lidar_points' is in config for dummy data generation
                num_lidar_points = self.config['data'].get('lidar_points', 1024)
                return torch.randn(num_lidar_points, 3) # Dummy LiDAR points (N, 3)
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
            num_lidar_points = self.config['data'].get('lidar_points', 1024)
            return torch.randn(num_lidar_points, 3)
        elif sensor_type == 'radar':
            # Example: np.fromfile(path, dtype=np.float32).reshape(...)
            return torch.randn(1, 64, 64)
        else:
            raise ValueError(f"Unknown sensor type: {sensor_type}")

    def _apply_augmentation(self, camera, lidar, radar):
        """Applies data augmentation to the sensor data.

        This is a placeholder for actual augmentation techniques such as random cropping,
        color jittering for camera images, or random rotations/scaling for LiDAR/radar data.

        Args:
            camera (torch.Tensor): Camera sensor data.
            lidar (torch.Tensor): LiDAR sensor data.
            radar (torch.Tensor): Radar sensor data.

        Returns:
            tuple: A tuple containing the augmented camera, lidar, and radar tensors.
        """
        if self.data_augmentation:
            # Example: RandomCrop, ColorJitter for camera; random rotation for lidar/radar
            camera = camera + torch.randn_like(camera) * 0.01
            lidar = lidar + torch.randn_like(lidar) * 0.01
            radar = radar + torch.randn_like(radar) * 0.01
        return camera, lidar, radar

    def __getitem__(self, idx):
        """Retrieves a single multi-modal data sample from the dataset.

        Args:
            idx (int): The index of the sample to retrieve.

        Returns:
            dict: A dictionary containing the multi-modal sensor data, strategic and
                  tactical actions, and the sample ID.
                  - 'camera' (torch.Tensor): Camera image data.
                  - 'lidar' (torch.Tensor): LiDAR point cloud data.
                  - 'radar' (torch.Tensor): Radar data.
                  - 'strategic_action' (torch.Tensor): Strategic action label (long).
                  - 'tactical_action' (torch.Tensor): Tactical action vector (float32).
                  - 'sample_id' (str): Unique identifier for the sample.
        """
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

    Args:
        batch (list): A list of dictionaries, where each dictionary is a sample
                      returned by MultiModalDrivingDataset.__getitem__.

    Returns:
        dict: A dictionary containing batched tensors for each modality and action,
              and a list of sample IDs.
              - 'camera' (torch.Tensor): Batched camera images (B, C, H, W).
              - 'lidar' (torch.Tensor): Batched and padded LiDAR point clouds (B, N_max, 3).
              - 'radar' (torch.Tensor): Batched radar data (B, C, H, W).
              - 'strategic_action' (torch.Tensor): Batched strategic actions (B,).
              - 'tactical_action' (torch.Tensor): Batched tactical actions (B, 3).
              - 'sample_id' (list): List of sample IDs for the batch.
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

    This function initializes the dataset and wraps it in a DataLoader,
    applying a custom collate function to handle variable-sized LiDAR data.

    Args:
        config (dict): Configuration dictionary containing dataset parameters.
                       Expected to have a 'data' key with 'batch_size', 'num_workers',
                       and optionally 'augmentation', 'num_samples', 'lidar_points'.
        split (str): Dataset split, e.g., 'train', 'val', 'test'. Defaults to 'train'.

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
