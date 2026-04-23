"""
Acoustic Dataset Preparation for Training

This module implements the merged audio approach for acoustic feature extraction,
where multiple audio files are concatenated together to create more robust
training samples for the acoustic/prosodic component.

The approach:
1. Groups audio files by diagnosis (ASD/TD) from multiple datasets
2. Creates merged samples by concatenating n_per_group audio files
3. Limits the total number of merged samples per diagnosis to prevent imbalance
4. Ensures reproducible sampling with fixed random seed

Author: ASD Detection Team
Date: March 2026
"""

import random
from pathlib import Path
from typing import List, Tuple, Optional, Dict
import numpy as np

from src.utils.logger import get_logger

logger = get_logger(__name__)


def prepare_acoustic_training_data(
    dataset_paths: List[Path],
    n_per_group: int = 20,  # TD: merge 20 files per group (updated from 10)
    max_merged_per_diagnosis: int = 100,  # TD: Create 100 merged training samples
    random_state: int = 42,
    min_audio_duration: float = 1.0,
    max_audio_duration: float = 300.0
) -> List[Tuple[List[Path], str, str]]:
    """
    Prepare acoustic training data with different strategies for ASD and TD.

    Strategy:
    - ASD files: Individual processing (no merging, just child-only audio extraction)
    - TD files: Merge 20 files into 1 group, create 100 such merged groups randomly

    This approach handles the class imbalance where TD datasets are much larger than ASD.

    Args:
        dataset_paths: List of dataset directory paths to scan
        n_per_group: Number of TD files to merge per training sample (default: 20)
        max_merged_per_diagnosis: Maximum merged groups for TD (default: 100)
        random_state: Random seed for reproducible sampling
        min_audio_duration: Minimum audio duration in seconds (for filtering)
        max_audio_duration: Maximum audio duration in seconds (for filtering)

    Returns:
        List of (audio_paths_list, diagnosis, dataset_name) tuples where:
        - ASD entries have 1 file per list (individual processing)
        - TD entries have 20 files per list (merged processing)

    Example:
        >>> paths = [Path("data/asdbank_eigsti"), Path("data/td")]
        >>> prepared = prepare_acoustic_training_data(paths)
        >>> # Returns: [([asd1.wav], 'ASD', 'eigsti'), ([td1.wav, td2.wav, ...20 files], 'TD', 'td'), ...]
    """
    logger.info(f"Preparing acoustic training data from {len(dataset_paths)} datasets")
    logger.info(f"Config: n_per_group={n_per_group}, max_per_diagnosis={max_merged_per_diagnosis}")

    # Set random seed for reproducibility
    random.seed(random_state)
    np.random.seed(random_state)

    # Collect all audio files by diagnosis
    audio_files_by_diagnosis: Dict[str, List[Tuple[Path, str]]] = {
        'ASD': [],
        'TD': []
    }

    # Scan all dataset paths
    for dataset_path in dataset_paths:
        if not dataset_path.exists():
            logger.warning(f"Dataset path not found: {dataset_path}")
            continue

        dataset_name = dataset_path.name
        logger.info(f"Scanning dataset: {dataset_name}")

        # Find audio files
        audio_files = _find_audio_files(dataset_path)
        logger.info(f"Found {len(audio_files)} audio files in {dataset_name}")

        # Classify files by diagnosis
        asd_found = 0
        td_found = 0
        unknown_found = 0

        for audio_file in audio_files:
            diagnosis = _infer_diagnosis(audio_file)

            if diagnosis == 'ASD':
                asd_found += 1
                audio_files_by_diagnosis[diagnosis].append((audio_file, dataset_name))
                logger.debug(f"ASD file found: {audio_file.name}")
            elif diagnosis == 'TD':
                td_found += 1
                audio_files_by_diagnosis[diagnosis].append((audio_file, dataset_name))
                logger.debug(f"TD file found: {audio_file.name}")
            elif diagnosis in audio_files_by_diagnosis:
                audio_files_by_diagnosis[diagnosis].append((audio_file, dataset_name))
            else:
                unknown_found += 1
                logger.debug(f"Unknown diagnosis for: {audio_file.name} (path: {audio_file})")

        logger.info(f"Dataset {dataset_name} classification: ASD={asd_found}, TD={td_found}, Unknown={unknown_found}")

    # Log statistics
    asd_count = len(audio_files_by_diagnosis['ASD'])
    td_count = len(audio_files_by_diagnosis['TD'])
    logger.info(f"Collected files - ASD: {asd_count}, TD: {td_count}")

    if asd_count == 0 and td_count == 0:
        logger.warning("No audio files found with recognizable diagnosis labels")
        return []

    # Create prepared groups based on diagnosis strategy
    prepared_groups = []

    # Strategy 1: ASD - Individual files (no merging, just child-only extraction)
    asd_files_with_datasets = audio_files_by_diagnosis['ASD']
    if asd_files_with_datasets:
        valid_asd_files = _filter_valid_audio_files(asd_files_with_datasets, min_audio_duration, max_audio_duration)
        logger.info(f"Valid ASD files after filtering: {len(valid_asd_files)}")

        if valid_asd_files:
            # For ASD: each file becomes its own "group" (no merging)
            for asd_file, dataset_name in valid_asd_files:
                # Single file per group for ASD
                prepared_groups.append(([asd_file], 'ASD', dataset_name))

            logger.info(f"Created {len(valid_asd_files)} individual ASD samples (no merging)")

    # Strategy 2: TD - Merged files (10 files per group, 100 groups total)
    td_files_with_datasets = audio_files_by_diagnosis['TD']
    if td_files_with_datasets:
        valid_td_files = _filter_valid_audio_files(td_files_with_datasets, min_audio_duration, max_audio_duration)
        logger.info(f"Valid TD files after filtering: {len(valid_td_files)}")

        if len(valid_td_files) < n_per_group:
            logger.warning(f"Not enough TD files ({len(valid_td_files)}) to create groups of {n_per_group}")
        else:
            # Shuffle TD files for random grouping
            shuffled_td_files = valid_td_files.copy()
            random.shuffle(shuffled_td_files)

            # Create merged groups for TD (10 files per group, up to 100 groups)
            td_groups_created = 0
            for i in range(0, len(shuffled_td_files) - n_per_group + 1, n_per_group):
                if td_groups_created >= max_merged_per_diagnosis:
                    break

                # Take next n_per_group files
                group_files_with_datasets = shuffled_td_files[i:i + n_per_group]
                group_files = [f[0] for f in group_files_with_datasets]  # Extract just the paths

                # Use the most common dataset name for this group
                dataset_names = [f[1] for f in group_files_with_datasets]
                most_common_dataset = max(set(dataset_names), key=dataset_names.count)

                # Add to prepared groups
                prepared_groups.append((group_files, 'TD', most_common_dataset))
                td_groups_created += 1

                logger.debug(f"Created TD merged group {td_groups_created}: {len(group_files)} files from {most_common_dataset}")

            logger.info(f"Created {td_groups_created} merged TD groups ({n_per_group} files per group)")

    logger.info(f"Total prepared groups: {len(prepared_groups)} (ASD individual + TD merged)")
    return prepared_groups


def _find_audio_files(dataset_path: Path) -> List[Path]:
    """Find all audio files in dataset directory."""
    audio_files = []
    audio_extensions = ['*.wav', '*.mp3', '*.flac', '*.m4a', '*.aac']

    for extension in audio_extensions:
        for audio_file in dataset_path.rglob(extension):
            # Skip files in child_only folders (from old extraction scripts)
            if 'child_only' in str(audio_file):
                continue
            # Skip hidden files and temp files
            if audio_file.name.startswith('.') or audio_file.name.startswith('~'):
                continue
            audio_files.append(audio_file)

    return audio_files


def _infer_diagnosis(audio_path: Path) -> Optional[str]:
    """
    Infer diagnosis from file path structure.

    Args:
        audio_path: Path to audio file

    Returns:
        'ASD', 'TD', or None if can't determine
    """
    path_str = str(audio_path).upper()

    # Check for explicit ASD indicators in path
    if any(indicator in path_str for indicator in ['/ASD/', '_ASD_', '\\ASD\\', '/ASD\\', '\\ASD/']):
        return 'ASD'

    # Check for explicit TD indicators in path
    td_indicators = ['/TD/', '/TYP/', '_TD_', '_TYP_', '\\TD\\', '\\TYP\\', '/TD\\', '/TYP\\', '\\TD/', '\\TYP/']
    if any(indicator in path_str for indicator in td_indicators):
        return 'TD'

    # Check for typical_dev or similar in path
    if 'TYPICAL' in path_str or 'NORMAL' in path_str or 'CONTROL' in path_str:
        return 'TD'

    # Check parent directory names for explicit indicators
    for parent in audio_path.parents:
        parent_name = parent.name.upper()
        if parent_name in ['ASD', 'AUTISM']:
            return 'ASD'
        elif parent_name in ['TD', 'TYP', 'TYPICAL', 'NORMAL', 'CONTROL']:
            return 'TD'

    # Dataset-specific inference for known ASD datasets
    dataset_parts = audio_path.parts
    for part in dataset_parts:
        part_upper = part.upper()

        # Check for known TD-only datasets by folder name
        if part_upper in ('OCSC', 'RESCORLA-TD'):
            return 'TD'

        # Check for known ASD dataset names
        if any(asd_name in part_upper for asd_name in ['ASDBANK_AAC', 'ASDBANK_EIGSTI', 'ASDBANK_FLUSBERG',
                                                       'ASDBANK_NADIG', 'ASDBANK_QUIGLEY', 'ASDBANK_ROLLINS']):
            # These are ASD datasets - classify as ASD unless explicitly TD
            if not any(td_indicator in path_str for td_indicator in ['/TD/', '\\TD\\', 'TYPICAL', 'CONTROL']):
                return 'ASD'

        # Check for general ASDBANK pattern
        elif 'ASDBANK' in part_upper:
            # Any ASDBANK dataset should be ASD unless explicitly TD
            if not any(td_indicator in path_str for td_indicator in ['/TD/', '\\TD\\', 'TYPICAL', 'CONTROL']):
                return 'ASD'

        # Check for TD dataset
        elif part_upper == 'TD' or 'TYPICAL' in part_upper:
            return 'TD'

        # Explicit ASD directory
        elif part_upper == 'ASD':
            return 'ASD'

    return None


def _filter_valid_audio_files(
    files_with_datasets: List[Tuple[Path, str]],
    min_duration: float,
    max_duration: float
) -> List[Tuple[Path, str]]:
    """
    Filter audio files by duration and validity.

    This is a basic implementation that checks file existence and basic size validation.
    For now, we use a permissive approach since actual duration checking would be too slow.

    Args:
        files_with_datasets: List of (audio_path, dataset_name) tuples
        min_duration: Minimum duration in seconds
        max_duration: Maximum duration in seconds

    Returns:
        Filtered list of valid files
    """
    valid_files = []

    for audio_path, dataset_name in files_with_datasets:
        # Basic validation: file exists
        if not audio_path.exists():
            logger.debug(f"Skipping {audio_path.name}: file does not exist")
            continue

        # Check file size - use more permissive range
        try:
            file_size_bytes = audio_path.stat().st_size

            # Very basic size check:
            # - Too small: < 1KB (likely corrupt)
            # - Too large: > 100MB (likely not a normal speech audio file)
            if file_size_bytes < 1024:  # Less than 1KB
                logger.debug(f"Skipping {audio_path.name}: file too small ({file_size_bytes} bytes)")
                continue
            elif file_size_bytes > 100 * 1024 * 1024:  # More than 100MB
                logger.debug(f"Skipping {audio_path.name}: file too large ({file_size_bytes / (1024*1024):.1f} MB)")
                continue

            # If we get here, the file passes basic validation
            valid_files.append((audio_path, dataset_name))

        except (OSError, PermissionError) as e:
            logger.debug(f"Skipping {audio_path.name}: cannot access file ({e})")
            continue

    logger.debug(f"Audio validation: {len(valid_files)}/{len(files_with_datasets)} files passed basic checks")
    return valid_files


def get_merged_sample_info(prepared_groups: List[Tuple[List[Path], str, str]]) -> Dict:
    """
    Get summary information about prepared merged samples.

    Args:
        prepared_groups: Output from prepare_acoustic_training_data

    Returns:
        Dictionary with summary statistics
    """
    if not prepared_groups:
        return {"total_groups": 0, "asd_groups": 0, "td_groups": 0, "files_per_group": 0}

    asd_groups = sum(1 for _, diagnosis, _ in prepared_groups if diagnosis == 'ASD')
    td_groups = sum(1 for _, diagnosis, _ in prepared_groups if diagnosis == 'TD')

    files_per_group = len(prepared_groups[0][0]) if prepared_groups else 0
    total_files = sum(len(files) for files, _, _ in prepared_groups)

    datasets = set(dataset for _, _, dataset in prepared_groups)

    return {
        "total_groups": len(prepared_groups),
        "asd_groups": asd_groups,
        "td_groups": td_groups,
        "files_per_group": files_per_group,
        "total_files_used": total_files,
        "datasets_involved": list(datasets)
    }


# Export main function
__all__ = ["prepare_acoustic_training_data", "get_merged_sample_info"]




