import unittest
from unittest.mock import MagicMock, patch
import os
import sys
import numpy as np

# --- Mock torch and PIL.Image before importing the dataset module ---
# This ensures that the dataset module imports the mocked versions.

# Create a mock for the entire torch module
mock_torch = MagicMock()

# Define a helper function to create mock tensors with shape and dtype
def create_mock_tensor(shape, dtype=None):
    mock_t = MagicMock()
    mock_t.shape = shape
    mock_t.dtype = dtype if dtype is not None else MagicMock()
    mock_t.cpu.return_value = mock_t # Mock .cpu() method
    mock_t.float.return_value = mock_t # Mock .float() method
    mock_t.long.return_value = mock_t # Mock .long() method
    return mock_t

# Configure the global mock_torch functions to return mock tensors with expected attributes
mock_torch.Tensor = MagicMock()
mock_torch.randn.side_effect = lambda *args, **kwargs: create_mock_tensor(args[0] if args else kwargs.get("size"), kwargs.get("dtype"))
mock_torch.tensor.side_effect = lambda x, dtype=None: create_mock_tensor(np.array(x).shape, dtype)
mock_torch.stack.side_effect = lambda x: create_mock_tensor((len(x),) + x[0].shape, x[0].dtype)
mock_torch.zeros.side_effect = lambda *args, **kwargs: create_mock_tensor(args[0] if args else kwargs.get("size"), kwargs.get("dtype"))
mock_torch.cat.side_effect = lambda x, dim: create_mock_tensor((x[0].shape[0] + x[1].shape[0], x[0].shape[1]) if dim == 0 else (x[0].shape[0], x[0].shape[1] + x[1].shape[1]), x[0].dtype)

# Mock torch dtypes
mock_torch.long = MagicMock()
mock_torch.float32 = MagicMock()

# Mock torch.utils.data.Dataset and DataLoader
mock_dataset_base = MagicMock()
mock_dataloader_base = MagicMock()

# Set up mock_torch.utils.data as a MagicMock that contains Dataset and DataLoader
mock_torch.utils = MagicMock()
mock_torch.utils.data = MagicMock()
mock_torch.utils.data.Dataset = mock_dataset_base
mock_torch.utils.data.DataLoader = mock_dataloader_base

# Mock PIL.Image
mock_pil_image = MagicMock()

# Use patch.dict to temporarily modify sys.modules for the import
with patch.dict("sys.modules", {
    "torch": mock_torch,
    "torch.utils": mock_torch.utils,
    "torch.utils.data": mock_torch.utils.data,
    "PIL.Image": mock_pil_image
}):
    # Add src to path for module import *inside* the patch.dict context
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
    from data.dataset import MultiModalDrivingDataset, collate_fn, get_dataloader

class TestMultiModalDrivingDataset(unittest.TestCase):

    def setUp(self):
        # Mock configuration for testing
        self.mock_config = {
            'data': {
                'batch_size': 2,
                'num_workers': 0, # Set to 0 for easier debugging in tests
                'augmentation': True,
                'num_samples': 5,
                'lidar_points': 1024 # Required for dummy data generation
            }
        }
        self.dataset_root = '/tmp/test_dataset_root'
        os.makedirs(self.dataset_root, exist_ok=True)

        # Reset mocks for each test to ensure isolation
        mock_torch.reset_mock()
        mock_pil_image.reset_mock()

        # Re-assign side_effects for torch mocks as reset_mock might clear them
        mock_torch.randn.side_effect = lambda *args, **kwargs: create_mock_tensor(args[0] if args else kwargs.get("size"), kwargs.get("dtype"))
        mock_torch.tensor.side_effect = lambda x, dtype=None: create_mock_tensor(np.array(x).shape, dtype)
        mock_torch.stack.side_effect = lambda x: create_mock_tensor((len(x),) + x[0].shape, x[0].dtype)
        mock_torch.zeros.side_effect = lambda *args, **kwargs: create_mock_tensor(args[0] if args else kwargs.get("size"), kwargs.get("dtype"))
        mock_torch.cat.side_effect = lambda x, dim: create_mock_tensor((x[0].shape[0] + x[1].shape[0], x[0].shape[1]) if dim == 0 else (x[0].shape[0], x[0].shape[1] + x[1].shape[1]), x[0].dtype)


    def tearDown(self):
        # Clean up dummy dataset root if created
        if os.path.exists(self.dataset_root):
            os.rmdir(self.dataset_root)

    def test_dataset_initialization(self):
        dataset = MultiModalDrivingDataset(self.mock_config, dataset_root=self.dataset_root)
        self.assertIsInstance(dataset, MultiModalDrivingDataset)
        self.assertEqual(len(dataset), self.mock_config['data']['num_samples'])
        self.assertTrue(dataset.data_augmentation)

    def test_generate_dummy_metadata(self):
        dataset = MultiModalDrivingDataset(self.mock_config, dataset_root=self.dataset_root)
        metadata = dataset._generate_dummy_metadata(num_samples=3)
        self.assertEqual(len(metadata), 3)
        self.assertIn('sample_id', metadata[0])
        self.assertIn('camera_path', metadata[0])
        self.assertIn('strategic_action', metadata[0])

    @patch('os.path.exists', return_value=False)
    def test_load_sensor_data_dummy_generation(self, mock_exists):
        dataset = MultiModalDrivingDataset(self.mock_config, dataset_root=self.dataset_root)
        
        # Test camera dummy generation
        camera_data = dataset._load_sensor_data('/nonexistent/path.jpg', 'camera')
        self.assertIsInstance(camera_data, mock_torch.Tensor)
        self.assertEqual(camera_data.shape, (3, 224, 224))

        # Test lidar dummy generation
        lidar_data = dataset._load_sensor_data('/nonexistent/path.bin', 'lidar')
        self.assertIsInstance(lidar_data, mock_torch.Tensor)
        self.assertEqual(lidar_data.shape, (self.mock_config['data']['lidar_points'], 3))

        # Test radar dummy generation
        radar_data = dataset._load_sensor_data('/nonexistent/path.pcd', 'radar')
        self.assertIsInstance(radar_data, mock_torch.Tensor)
        self.assertEqual(radar_data.shape, (1, 64, 64))

        # Test unknown sensor type
        with self.assertRaises(ValueError):
            dataset._load_sensor_data('/nonexistent/path.xyz', 'unknown')

    def test_getitem(self):
        dataset = MultiModalDrivingDataset(self.mock_config, dataset_root=self.dataset_root)
        sample = dataset[0]

        self.assertIn('camera', sample)
        self.assertIn('lidar', sample)
        self.assertIn('radar', sample)
        self.assertIn('strategic_action', sample)
        self.assertIn('tactical_action', sample)
        self.assertIn('sample_id', sample)

        self.assertIsInstance(sample['camera'], mock_torch.Tensor)
        self.assertIsInstance(sample['lidar'], mock_torch.Tensor)
        self.assertIsInstance(sample['radar'], mock_torch.Tensor)
        self.assertIsInstance(sample['strategic_action'], mock_torch.Tensor)
        self.assertIsInstance(sample['tactical_action'], mock_torch.Tensor)
        self.assertIsInstance(sample['sample_id'], str)

        self.assertEqual(sample['camera'].shape, (3, 224, 224))
        self.assertEqual(sample['lidar'].shape, (self.mock_config['data']['lidar_points'], 3))
        self.assertEqual(sample['radar'].shape, (1, 64, 64))
        self.assertEqual(sample['strategic_action'].dtype, mock_torch.long)
        self.assertEqual(sample['tactical_action'].dtype, mock_torch.float32)

    def test_collate_fn(self):
        dataset = MultiModalDrivingDataset(self.mock_config, dataset_root=self.dataset_root)
        # Create a batch of samples
        batch = [dataset[i] for i in range(self.mock_config['data']['num_samples'])] # Use num_samples for batch creation
        collated_batch = collate_fn(batch)

        self.assertIn('camera', collated_batch)
        self.assertIn('lidar', collated_batch)
        self.assertIn('radar', collated_batch)
        self.assertIn('strategic_action', collated_batch)
        self.assertIn('tactical_action', collated_batch)
        self.assertIn('sample_id', collated_batch)

        self.assertEqual(collated_batch['camera'].shape[0], self.mock_config['data']['num_samples'])
        self.assertEqual(collated_batch['lidar'].shape[0], self.mock_config['data']['num_samples'])
        self.assertEqual(collated_batch['radar'].shape[0], self.mock_config['data']['num_samples'])
        self.assertEqual(collated_batch['strategic_action'].shape[0], self.mock_config['data']['num_samples'])
        self.assertEqual(collated_batch['tactical_action'].shape[0], self.mock_config['data']['num_samples'])
        self.assertEqual(len(collated_batch['sample_id']), self.mock_config['data']['num_samples'])

        # Check LiDAR padding: all LiDAR samples in the batch should have the same number of points
        # which is the max_points from the original (dummy) samples.
        # Since dummy data has fixed lidar_points, max_points will be lidar_points
        self.assertEqual(collated_batch['lidar'].shape[1], self.mock_config['data']['lidar_points'])

    def test_get_dataloader(self):
        dataloader = get_dataloader(self.mock_config, split='train')
        self.assertIsInstance(dataloader, mock_dataloader_base)
        mock_dataloader_base.assert_called_once_with(
            unittest.mock.ANY, # MultiModalDrivingDataset instance
            batch_size=self.mock_config['data']['batch_size'],
            shuffle=True,
            num_workers=self.mock_config['data']['num_workers'],
            collate_fn=collate_fn
        )

        # Simulate iterating through dataloader
        mock_dataloader_instance = mock_dataloader_base.return_value
        mock_dataloader_instance.__iter__.return_value = [
            {
                'camera': mock_torch.randn(3, 224, 224),
                'lidar': mock_torch.randn(self.mock_config['data']['lidar_points'], 3),
                'radar': mock_torch.randn(1, 64, 64),
                'strategic_action': mock_torch.tensor(0, dtype=mock_torch.long),
                'tactical_action': mock_torch.tensor([0.1, 0.2, 0.3], dtype=mock_torch.float32),
                'sample_id': 'sample_000000'
            }
        ] * self.mock_config['data']['batch_size'] # Simulate a batch

        batch_count = 0
        for batch in dataloader:
            batch_count += 1
            self.assertIn('camera', batch)
            self.assertIn('lidar', batch)
            self.assertIn('radar', batch)
            self.assertIn('strategic_action', batch)
            self.assertIn('tactical_action', batch)
            self.assertIn('sample_id', batch)
            self.assertEqual(batch['camera'].shape, (3, 224, 224))
            self.assertEqual(batch['lidar'].shape, (self.mock_config['data']['lidar_points'], 3))
            self.assertEqual(batch['radar'].shape, (1, 64, 64))

        self.assertEqual(batch_count, 1)

if __name__ == '__main__':
    unittest.main()
