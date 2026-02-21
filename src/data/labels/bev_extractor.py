"""BEV (Bird's Eye View) semantic segmentation label extraction."""
import numpy as np
from typing import Dict, List, Tuple, Optional
from PIL import Image, ImageDraw
from pyquaternion import Quaternion


class BEVLabelExtractor:
    """Extract BEV semantic segmentation labels from nuScenes map data."""

    def __init__(
        self,
        nusc,
        bev_size: Tuple[int, int] = (200, 200),
        bev_range: float = 50.0,
        classes: Optional[Dict[int, str]] = None
    ):
        """
        Initialize BEV label extractor.

        Args:
            nusc: NuScenes instance
            bev_size: BEV image size in pixels (height, width)
            bev_range: BEV range in meters (centered on ego vehicle)
            classes: Dictionary mapping class index to class name
        """
        self.nusc = nusc
        self.bev_size = bev_size
        self.bev_range = bev_range

        # Default class mapping
        if classes is None:
            self.classes = {
                0: 'background',
                1: 'drivable_area',
                2: 'lane_divider',
                3: 'pedestrian_crossing',
                4: 'vehicle',
                5: 'pedestrian'
            }
        else:
            self.classes = classes

        self.num_classes = len(self.classes)

        # Meters per pixel
        self.resolution = (2 * bev_range) / bev_size[0]

    def extract_bev_labels(self, sample_token: str) -> np.ndarray:
        """
        Extract BEV semantic segmentation labels.

        Args:
            sample_token: Sample token to extract labels from

        Returns:
            BEV segmentation mask of shape (H, W) with class indices
        """
        sample = self.nusc.get('sample', sample_token)

        # Get ego pose
        lidar_token = sample['data']['LIDAR_TOP']
        lidar_data = self.nusc.get('sample_data', lidar_token)
        ego_pose_token = lidar_data['ego_pose_token']
        ego_pose = self.nusc.get('ego_pose', ego_pose_token)

        ego_translation = np.array(ego_pose['translation'])
        ego_rotation = Quaternion(ego_pose['rotation'])

        # Initialize BEV canvas
        bev_mask = np.zeros(self.bev_size, dtype=np.uint8)

        # Get map name for this sample
        scene = self.nusc.get('scene', sample['scene_token'])
        log = self.nusc.get('log', scene['log_token'])
        map_name = log['location']

        # Check if map API is available
        try:
            from nuscenes.map_expansion.map_api import NuScenesMap
            nusc_map = NuScenesMap(dataroot=self.nusc.dataroot, map_name=map_name)
        except Exception:
            # Map not available, return empty mask
            return bev_mask

        # Extract map layers
        bev_mask = self._rasterize_map_layers(
            nusc_map,
            ego_translation,
            ego_rotation,
            bev_mask
        )

        # Add dynamic objects (vehicles, pedestrians)
        bev_mask = self._add_dynamic_objects(
            sample,
            ego_translation,
            ego_rotation,
            bev_mask
        )

        return bev_mask

    def _rasterize_map_layers(
        self,
        nusc_map,
        ego_translation: np.ndarray,
        ego_rotation: Quaternion,
        bev_mask: np.ndarray
    ) -> np.ndarray:
        """
        Rasterize map layers to BEV grid.

        Args:
            nusc_map: NuScenesMap instance
            ego_translation: Ego vehicle position in global frame
            ego_rotation: Ego vehicle rotation
            bev_mask: BEV mask to update

        Returns:
            Updated BEV mask with map layers
        """
        # Layer names in nuScenes map
        layer_names = [
            'drivable_area',
            'lane_divider',
            'ped_crossing'
        ]

        # Get patch box around ego vehicle
        patch_box = (
            ego_translation[0] - self.bev_range,
            ego_translation[1] - self.bev_range,
            ego_translation[0] + self.bev_range,
            ego_translation[1] + self.bev_range
        )

        # Extract each layer
        for layer_name in layer_names:
            try:
                # Get records in patch
                records = nusc_map.get_records_in_patch(
                    patch_box,
                    [layer_name],
                    mode='intersect'
                )

                if layer_name not in records or not records[layer_name]:
                    continue

                # Get class index for this layer
                class_idx = self._get_class_index_for_layer(layer_name)

                # Rasterize each polygon/polyline
                for record_token in records[layer_name]:
                    self._rasterize_record(
                        nusc_map,
                        layer_name,
                        record_token,
                        ego_translation,
                        ego_rotation,
                        bev_mask,
                        class_idx
                    )

            except Exception:
                # Layer not available or error, skip
                continue

        return bev_mask

    def _rasterize_record(
        self,
        nusc_map,
        layer_name: str,
        record_token: str,
        ego_translation: np.ndarray,
        ego_rotation: Quaternion,
        bev_mask: np.ndarray,
        class_idx: int
    ):
        """
        Rasterize a single map record to BEV.

        Args:
            nusc_map: NuScenesMap instance
            layer_name: Layer name
            record_token: Record token
            ego_translation: Ego position
            ego_rotation: Ego rotation
            bev_mask: BEV mask to update
            class_idx: Class index for this record
        """
        try:
            # Get polygon/polyline
            if layer_name == 'drivable_area':
                polygon = nusc_map.extract_polygon(record_token)
                exterior_coords = list(polygon.exterior.coords)
                self._draw_polygon(
                    exterior_coords,
                    ego_translation,
                    ego_rotation,
                    bev_mask,
                    class_idx
                )

            elif layer_name in ['lane_divider', 'ped_crossing']:
                # These are polylines
                line = nusc_map.extract_line(record_token)
                coords = list(line.coords)
                self._draw_polyline(
                    coords,
                    ego_translation,
                    ego_rotation,
                    bev_mask,
                    class_idx,
                    width=2  # pixels
                )

        except Exception:
            # Skip problematic records
            pass

    def _draw_polygon(
        self,
        coords: List[Tuple[float, float]],
        ego_translation: np.ndarray,
        ego_rotation: Quaternion,
        bev_mask: np.ndarray,
        class_idx: int
    ):
        """
        Draw filled polygon on BEV mask.

        Args:
            coords: List of (x, y) coordinates in global frame
            ego_translation: Ego position
            ego_rotation: Ego rotation
            bev_mask: BEV mask to update
            class_idx: Class index to fill
        """
        # Transform coordinates to ego frame
        ego_coords = []
        for x, y in coords:
            global_pos = np.array([x, y])
            ego_pos = self._transform_to_ego_frame(
                global_pos,
                ego_translation[:2],
                ego_rotation
            )
            pixel = self._world_to_pixel(ego_pos)
            if pixel is not None:
                ego_coords.append(pixel)

        if len(ego_coords) < 3:
            return

        # Use PIL to draw polygon
        img = Image.fromarray(bev_mask)
        draw = ImageDraw.Draw(img)
        draw.polygon(ego_coords, fill=class_idx)
        bev_mask[:] = np.array(img)

    def _draw_polyline(
        self,
        coords: List[Tuple[float, float]],
        ego_translation: np.ndarray,
        ego_rotation: Quaternion,
        bev_mask: np.ndarray,
        class_idx: int,
        width: int = 1
    ):
        """
        Draw polyline on BEV mask.

        Args:
            coords: List of (x, y) coordinates in global frame
            ego_translation: Ego position
            ego_rotation: Ego rotation
            bev_mask: BEV mask to update
            class_idx: Class index
            width: Line width in pixels
        """
        # Transform coordinates to ego frame
        ego_coords = []
        for x, y in coords:
            global_pos = np.array([x, y])
            ego_pos = self._transform_to_ego_frame(
                global_pos,
                ego_translation[:2],
                ego_rotation
            )
            pixel = self._world_to_pixel(ego_pos)
            if pixel is not None:
                ego_coords.append(pixel)

        if len(ego_coords) < 2:
            return

        # Use PIL to draw line
        img = Image.fromarray(bev_mask)
        draw = ImageDraw.Draw(img)
        draw.line(ego_coords, fill=class_idx, width=width)
        bev_mask[:] = np.array(img)

    def _add_dynamic_objects(
        self,
        sample: Dict,
        ego_translation: np.ndarray,
        ego_rotation: Quaternion,
        bev_mask: np.ndarray
    ) -> np.ndarray:
        """
        Add dynamic objects (vehicles, pedestrians) to BEV mask.

        Args:
            sample: Sample record
            ego_translation: Ego position
            ego_rotation: Ego rotation
            bev_mask: BEV mask to update

        Returns:
            Updated BEV mask with dynamic objects
        """
        for ann_token in sample['anns']:
            ann = self.nusc.get('sample_annotation', ann_token)

            # Get object category
            category = ann['category_name']

            # Determine class index
            if 'vehicle' in category.lower():
                class_idx = self._get_class_index('vehicle')
            elif 'pedestrian' in category.lower() or 'human' in category.lower():
                class_idx = self._get_class_index('pedestrian')
            else:
                continue  # Skip other objects

            # Get object position and size
            obj_translation = np.array(ann['translation'])
            obj_rotation = Quaternion(ann['rotation'])
            obj_size = ann['size']  # [width, length, height]

            # Create bounding box corners in object frame
            w, l, h = obj_size[0], obj_size[1], obj_size[2]
            corners_obj = np.array([
                [-l/2, -w/2],
                [-l/2,  w/2],
                [ l/2,  w/2],
                [ l/2, -w/2]
            ])

            # Transform corners to global frame
            corners_global = []
            for corner in corners_obj:
                # Rotate
                corner_3d = np.array([corner[0], corner[1], 0.0])
                rotated = obj_rotation.rotate(corner_3d)
                # Translate
                global_pos = rotated[:2] + obj_translation[:2]
                corners_global.append(global_pos)

            # Transform to ego frame and convert to pixels
            corners_pixel = []
            for corner_global in corners_global:
                ego_pos = self._transform_to_ego_frame(
                    corner_global,
                    ego_translation[:2],
                    ego_rotation
                )
                pixel = self._world_to_pixel(ego_pos)
                if pixel is not None:
                    corners_pixel.append(pixel)

            if len(corners_pixel) < 3:
                continue

            # Draw filled polygon for object
            img = Image.fromarray(bev_mask)
            draw = ImageDraw.Draw(img)
            draw.polygon(corners_pixel, fill=class_idx)
            bev_mask[:] = np.array(img)

        return bev_mask

    def _transform_to_ego_frame(
        self,
        global_pos: np.ndarray,
        ego_translation: np.ndarray,
        ego_rotation: Quaternion
    ) -> np.ndarray:
        """
        Transform position from global frame to ego frame.

        Args:
            global_pos: Position in global frame (2,)
            ego_translation: Ego position in global frame (2,)
            ego_rotation: Ego rotation quaternion

        Returns:
            Position in ego frame (2,)
        """
        # Translate to ego origin
        relative_pos = global_pos - ego_translation

        # Rotate to ego frame (inverse rotation)
        relative_pos_3d = np.array([relative_pos[0], relative_pos[1], 0.0])
        rotated_3d = ego_rotation.inverse.rotate(relative_pos_3d)

        return rotated_3d[:2]

    def _world_to_pixel(self, pos: np.ndarray) -> Optional[Tuple[int, int]]:
        """
        Convert world coordinates (ego frame) to pixel coordinates.

        Args:
            pos: Position in ego frame (2,) in meters, with x forward, y left

        Returns:
            Pixel coordinates (u, v) or None if out of bounds
        """
        # Ego frame: x forward, y left
        # BEV image: u right (corresponds to -y), v down (corresponds to x)

        # Center of BEV is at (H/2, W/2)
        center_v = self.bev_size[0] / 2
        center_u = self.bev_size[1] / 2

        # Convert to pixel coordinates
        # x (forward) -> v (down from center)
        # y (left) -> -u (left from center, but u increases right)
        v = center_v + pos[0] / self.resolution
        u = center_u - pos[1] / self.resolution

        # Check bounds
        if 0 <= u < self.bev_size[1] and 0 <= v < self.bev_size[0]:
            return (int(u), int(v))
        else:
            return None

    def _get_class_index_for_layer(self, layer_name: str) -> int:
        """
        Get class index for a map layer.

        Args:
            layer_name: Map layer name

        Returns:
            Class index
        """
        mapping = {
            'drivable_area': 'drivable_area',
            'lane_divider': 'lane_divider',
            'ped_crossing': 'pedestrian_crossing'
        }

        class_name = mapping.get(layer_name, 'background')
        return self._get_class_index(class_name)

    def _get_class_index(self, class_name: str) -> int:
        """
        Get class index from class name.

        Args:
            class_name: Class name

        Returns:
            Class index (0 if not found)
        """
        for idx, name in self.classes.items():
            if name == class_name:
                return idx
        return 0  # Background
