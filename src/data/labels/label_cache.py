"""Label caching system for fast loading of extracted labels."""
import pickle
import os
from pathlib import Path
from typing import Dict, Optional
import time


class LabelCache:
    """Cache system for storing and loading extracted labels."""

    def __init__(self, cache_dir: str = './cache/labels'):
        """
        Initialize label cache.

        Args:
            cache_dir: Directory to store cached labels
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def save_labels(self, sample_token: str, labels: Dict, split: str = 'train'):
        """
        Save labels to cache.

        Args:
            sample_token: Sample token (unique identifier)
            labels: Labels dictionary to cache
            split: Dataset split ('train', 'val', 'test')
        """
        # Create split directory if needed
        split_dir = self.cache_dir / split
        split_dir.mkdir(parents=True, exist_ok=True)

        # Save to pickle file
        cache_file = split_dir / f"{sample_token}.pkl"

        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(labels, f, protocol=pickle.HIGHEST_PROTOCOL)
        except Exception as e:
            print(f"Warning: Failed to cache labels for {sample_token}: {e}")

    def load_labels(self, sample_token: str, split: str = 'train') -> Optional[Dict]:
        """
        Load labels from cache.

        Args:
            sample_token: Sample token to load
            split: Dataset split ('train', 'val', 'test')

        Returns:
            Cached labels dictionary or None if not found
        """
        cache_file = self.cache_dir / split / f"{sample_token}.pkl"

        if not cache_file.exists():
            return None

        try:
            with open(cache_file, 'rb') as f:
                labels = pickle.load(f)
            return labels
        except Exception as e:
            print(f"Warning: Failed to load cached labels for {sample_token}: {e}")
            return None

    def check_cache(self, sample_token: str, split: str = 'train') -> bool:
        """
        Check if labels are cached for a sample.

        Args:
            sample_token: Sample token to check
            split: Dataset split

        Returns:
            True if cached, False otherwise
        """
        cache_file = self.cache_dir / split / f"{sample_token}.pkl"
        return cache_file.exists()

    def clear_cache(self, split: Optional[str] = None):
        """
        Clear cached labels.

        Args:
            split: Specific split to clear, or None to clear all
        """
        if split is None:
            # Clear entire cache directory
            for split_dir in self.cache_dir.iterdir():
                if split_dir.is_dir():
                    for cache_file in split_dir.glob('*.pkl'):
                        cache_file.unlink()
            print(f"Cleared all cached labels from {self.cache_dir}")
        else:
            # Clear specific split
            split_dir = self.cache_dir / split
            if split_dir.exists():
                for cache_file in split_dir.glob('*.pkl'):
                    cache_file.unlink()
                print(f"Cleared cached labels for split: {split}")

    def get_cache_stats(self, split: str = 'train') -> Dict:
        """
        Get statistics about cached labels.

        Args:
            split: Dataset split to analyze

        Returns:
            Dictionary with cache statistics
        """
        split_dir = self.cache_dir / split

        if not split_dir.exists():
            return {
                'num_cached': 0,
                'total_size_mb': 0.0,
                'cache_dir': str(split_dir),
                'exists': False
            }

        # Count cached files
        cache_files = list(split_dir.glob('*.pkl'))
        num_cached = len(cache_files)

        # Calculate total size
        total_size_bytes = sum(f.stat().st_size for f in cache_files)
        total_size_mb = total_size_bytes / (1024 * 1024)

        return {
            'num_cached': num_cached,
            'total_size_mb': total_size_mb,
            'cache_dir': str(split_dir),
            'exists': True
        }

    def get_all_stats(self) -> Dict:
        """
        Get statistics for all splits.

        Returns:
            Dictionary mapping split -> stats
        """
        all_stats = {}

        # Check for common splits
        for split in ['train', 'val', 'test']:
            stats = self.get_cache_stats(split)
            if stats['exists'] or stats['num_cached'] > 0:
                all_stats[split] = stats

        return all_stats

    def print_cache_summary(self):
        """Print human-readable cache summary."""
        print("\n" + "="*60)
        print("Label Cache Summary")
        print("="*60)
        print(f"Cache directory: {self.cache_dir}")

        all_stats = self.get_all_stats()

        if not all_stats:
            print("No cached labels found.")
        else:
            total_cached = 0
            total_size = 0.0

            for split, stats in all_stats.items():
                print(f"\n{split.upper()} split:")
                print(f"  Cached samples:  {stats['num_cached']}")
                print(f"  Total size:      {stats['total_size_mb']:.2f} MB")

                total_cached += stats['num_cached']
                total_size += stats['total_size_mb']

            print(f"\nTotal:")
            print(f"  Cached samples:  {total_cached}")
            print(f"  Total size:      {total_size:.2f} MB")

        print("="*60 + "\n")

    def warm_cache(
        self,
        sample_tokens: list,
        extractor_fn,
        split: str = 'train',
        show_progress: bool = True
    ):
        """
        Pre-populate cache for a list of samples.

        Args:
            sample_tokens: List of sample tokens to cache
            extractor_fn: Function that takes sample_token and returns labels dict
            split: Dataset split
            show_progress: Whether to show progress bar
        """
        from tqdm import tqdm

        # Filter out already cached samples
        samples_to_extract = [
            token for token in sample_tokens
            if not self.check_cache(token, split)
        ]

        if not samples_to_extract:
            print(f"All {len(sample_tokens)} samples already cached.")
            return

        print(f"Caching labels for {len(samples_to_extract)}/{len(sample_tokens)} samples...")

        start_time = time.time()

        iterator = tqdm(samples_to_extract) if show_progress else samples_to_extract

        for token in iterator:
            try:
                # Extract labels
                labels = extractor_fn(token)

                # Save to cache
                self.save_labels(token, labels, split)

            except Exception as e:
                if show_progress:
                    iterator.write(f"Error caching {token}: {e}")
                else:
                    print(f"Error caching {token}: {e}")

        elapsed = time.time() - start_time
        print(f"Cached {len(samples_to_extract)} samples in {elapsed:.1f}s ({elapsed/len(samples_to_extract):.2f}s per sample)")

    def verify_cache_integrity(self, split: str = 'train') -> Dict:
        """
        Verify integrity of cached labels.

        Args:
            split: Dataset split to verify

        Returns:
            Dictionary with verification results
        """
        split_dir = self.cache_dir / split

        if not split_dir.exists():
            return {
                'total_files': 0,
                'valid_files': 0,
                'corrupted_files': 0,
                'corrupted_tokens': []
            }

        cache_files = list(split_dir.glob('*.pkl'))
        total_files = len(cache_files)
        valid_files = 0
        corrupted_files = 0
        corrupted_tokens = []

        for cache_file in cache_files:
            token = cache_file.stem

            try:
                # Try to load
                with open(cache_file, 'rb') as f:
                    pickle.load(f)
                valid_files += 1
            except Exception:
                corrupted_files += 1
                corrupted_tokens.append(token)

        return {
            'total_files': total_files,
            'valid_files': valid_files,
            'corrupted_files': corrupted_files,
            'corrupted_tokens': corrupted_tokens
        }
