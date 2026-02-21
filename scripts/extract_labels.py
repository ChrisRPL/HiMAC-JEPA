#!/usr/bin/env python3
"""Pre-extract ground truth labels for entire dataset."""
import argparse
import sys
from pathlib import Path
import time
from tqdm import tqdm

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.labels import TrajectoryLabelExtractor, BEVLabelExtractor, MotionLabelExtractor, LabelCache


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Extract ground truth labels from nuScenes')

    parser.add_argument(
        '--data-root',
        type=str,
        default='/data/nuscenes',
        help='Path to nuScenes dataset (default: /data/nuscenes)'
    )

    parser.add_argument(
        '--version',
        type=str,
        default='v1.0-mini',
        choices=['v1.0-mini', 'v1.0-trainval'],
        help='nuScenes version (default: v1.0-mini)'
    )

    parser.add_argument(
        '--split',
        type=str,
        default='train',
        choices=['train', 'val', 'test', 'all'],
        help='Dataset split to extract (default: train)'
    )

    parser.add_argument(
        '--cache-dir',
        type=str,
        default='./cache/labels',
        help='Cache directory (default: ./cache/labels)'
    )

    parser.add_argument(
        '--workers',
        type=int,
        default=1,
        help='Number of parallel workers (default: 1, 0 for serial)'
    )

    parser.add_argument(
        '--force',
        action='store_true',
        help='Force recomputation even if cached'
    )

    parser.add_argument(
        '--trajectory-horizons',
        nargs='+',
        type=float,
        default=[1.0, 2.0, 3.0],
        help='Trajectory prediction horizons in seconds (default: 1.0 2.0 3.0)'
    )

    parser.add_argument(
        '--bev-size',
        nargs=2,
        type=int,
        default=[200, 200],
        help='BEV image size (default: 200 200)'
    )

    parser.add_argument(
        '--bev-range',
        type=float,
        default=50.0,
        help='BEV range in meters (default: 50.0)'
    )

    parser.add_argument(
        '--motion-horizon',
        type=float,
        default=3.0,
        help='Motion prediction horizon in seconds (default: 3.0)'
    )

    parser.add_argument(
        '--max-distance',
        type=float,
        default=50.0,
        help='Maximum distance for agent tracking (default: 50.0)'
    )

    return parser.parse_args()


def extract_labels_for_sample(sample_token, nusc, extractors, label_cache, split, force):
    """
    Extract labels for a single sample.

    Args:
        sample_token: Sample token
        nusc: NuScenes instance
        extractors: Tuple of (traj_extractor, bev_extractor, motion_extractor)
        label_cache: LabelCache instance
        split: Dataset split
        force: Force recomputation

    Returns:
        True if successful, False otherwise
    """
    # Check cache first
    if not force and label_cache.check_cache(sample_token, split):
        return True

    traj_extractor, bev_extractor, motion_extractor = extractors

    try:
        # Extract all labels
        labels = {
            'trajectory_ego': traj_extractor.extract_ego_trajectory(sample_token),
            'trajectory_agents': traj_extractor.extract_agent_trajectories(sample_token),
            'bev': bev_extractor.extract_bev_labels(sample_token),
            'motion': motion_extractor.extract_motion_labels(sample_token)
        }

        # Save to cache
        label_cache.save_labels(sample_token, labels, split)

        return True

    except Exception as e:
        print(f"\nError extracting labels for {sample_token}: {e}")
        return False


def extract_for_split(args, nusc, split):
    """Extract labels for a specific split."""
    print(f"\n{'='*60}")
    print(f"Extracting labels for {split.upper()} split")
    print(f"{'='*60}")

    # Initialize extractors
    print("Initializing extractors...")
    traj_extractor = TrajectoryLabelExtractor(
        nusc,
        pred_horizons=args.trajectory_horizons
    )

    bev_extractor = BEVLabelExtractor(
        nusc,
        bev_size=tuple(args.bev_size),
        bev_range=args.bev_range
    )

    motion_extractor = MotionLabelExtractor(
        nusc,
        pred_horizon=args.motion_horizon,
        max_distance=args.max_distance
    )

    # Initialize cache
    label_cache = LabelCache(cache_dir=args.cache_dir)

    # Get samples for this split
    from nuscenes.utils.splits import create_splits_scenes

    splits = create_splits_scenes()
    if split == 'test' and split not in splits:
        split = 'val'  # Fallback

    scene_names = set(splits.get(split, []))

    # Collect all sample tokens
    sample_tokens = []
    for scene in nusc.scene:
        if scene['name'] in scene_names:
            sample_token = scene['first_sample_token']
            while sample_token:
                sample_tokens.append(sample_token)
                sample = nusc.get('sample', sample_token)
                sample_token = sample.get('next', '')

    print(f"Found {len(sample_tokens)} samples in {split} split")

    # Check cache
    if not args.force:
        cached_count = sum(1 for token in sample_tokens
                          if label_cache.check_cache(token, split))
        print(f"Already cached: {cached_count}/{len(sample_tokens)}")

        if cached_count == len(sample_tokens):
            print("All samples already cached. Use --force to recompute.")
            return

    # Extract labels
    print("\nExtracting labels...")
    start_time = time.time()

    extractors = (traj_extractor, bev_extractor, motion_extractor)

    if args.workers > 1:
        # Parallel extraction
        from multiprocessing import Pool
        from functools import partial

        extract_fn = partial(
            extract_labels_for_sample,
            nusc=nusc,
            extractors=extractors,
            label_cache=label_cache,
            split=split,
            force=args.force
        )

        with Pool(processes=args.workers) as pool:
            results = list(tqdm(
                pool.imap(extract_fn, sample_tokens),
                total=len(sample_tokens),
                desc="Extracting"
            ))
    else:
        # Serial extraction
        results = []
        for token in tqdm(sample_tokens, desc="Extracting"):
            success = extract_labels_for_sample(
                token, nusc, extractors, label_cache, split, args.force
            )
            results.append(success)

    elapsed = time.time() - start_time

    # Print statistics
    successful = sum(results)
    failed = len(results) - successful

    print(f"\n{'='*60}")
    print(f"Extraction complete for {split.upper()} split")
    print(f"{'='*60}")
    print(f"Successful:     {successful}/{len(sample_tokens)}")
    print(f"Failed:         {failed}")
    print(f"Time elapsed:   {elapsed:.1f}s")
    print(f"Time per sample: {elapsed/len(sample_tokens):.2f}s")

    # Print cache stats
    cache_stats = label_cache.get_cache_stats(split)
    print(f"\nCache statistics:")
    print(f"  Cached samples: {cache_stats['num_cached']}")
    print(f"  Total size:     {cache_stats['total_size_mb']:.2f} MB")
    print(f"{'='*60}\n")


def main():
    """Main extraction function."""
    args = parse_args()

    print("="*60)
    print("nuScenes Ground Truth Label Extraction")
    print("="*60)
    print(f"Data root:      {args.data_root}")
    print(f"Version:        {args.version}")
    print(f"Split:          {args.split}")
    print(f"Cache dir:      {args.cache_dir}")
    print(f"Workers:        {args.workers}")
    print(f"Force recompute: {args.force}")
    print(f"\nLabel configuration:")
    print(f"  Trajectory horizons: {args.trajectory_horizons}")
    print(f"  BEV size:           {args.bev_size}")
    print(f"  BEV range:          {args.bev_range}m")
    print(f"  Motion horizon:     {args.motion_horizon}s")
    print(f"  Max distance:       {args.max_distance}m")
    print("="*60)

    # Check if nuScenes is available
    try:
        from nuscenes.nuscenes import NuScenes
    except ImportError:
        print("\nError: nuScenes devkit not installed.")
        print("Install with: pip install nuscenes-devkit")
        return 1

    # Load nuScenes
    print(f"\nLoading nuScenes {args.version}...")
    try:
        nusc = NuScenes(
            version=args.version,
            dataroot=args.data_root,
            verbose=True
        )
    except Exception as e:
        print(f"\nError loading nuScenes: {e}")
        print(f"Make sure {args.data_root} exists and contains {args.version}")
        return 1

    # Extract for requested splits
    if args.split == 'all':
        splits_to_extract = ['train', 'val']
    else:
        splits_to_extract = [args.split]

    total_start = time.time()

    for split in splits_to_extract:
        extract_for_split(args, nusc, split)

    total_elapsed = time.time() - total_start

    # Print overall summary
    print("\n" + "="*60)
    print("Overall Summary")
    print("="*60)

    label_cache = LabelCache(cache_dir=args.cache_dir)
    all_stats = label_cache.get_all_stats()

    total_cached = 0
    total_size = 0.0

    for split, stats in all_stats.items():
        print(f"\n{split.upper()}:")
        print(f"  Samples: {stats['num_cached']}")
        print(f"  Size:    {stats['total_size_mb']:.2f} MB")
        total_cached += stats['num_cached']
        total_size += stats['total_size_mb']

    print(f"\nTotal cached:    {total_cached} samples")
    print(f"Total size:      {total_size:.2f} MB")
    print(f"Total time:      {total_elapsed:.1f}s")
    print("="*60)

    return 0


if __name__ == '__main__':
    sys.exit(main())
