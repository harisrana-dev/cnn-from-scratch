"""Constants, validation and file I/O shared by the dataset pipeline.

This module contains no TensorFlow/Keras imports: every artefact produced for
the experiments can be inspected with NumPy alone, which is what the scratch
implementation needs.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any, Mapping

import numpy as np

# --------------------------------------------------------------------------- #
# Paths (resolved relative to the repository, never hard-coded absolutely)
# --------------------------------------------------------------------------- #
PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
FIGURES_DIR = PROJECT_ROOT / "results" / "figures"
CONFIG_PATH = PROJECT_ROOT / "configs" / "experiment_config.json"

DATASET_FILENAME = "fashion_mnist_subset.npz"
METADATA_FILENAME = "fashion_mnist_subset_metadata.json"
DATASET_PATH = PROCESSED_DIR / DATASET_FILENAME
METADATA_PATH = PROCESSED_DIR / METADATA_FILENAME

# --------------------------------------------------------------------------- #
# Dataset specification (Fashion-MNIST, https://github.com/zalandoresearch/fashion-mnist)
# --------------------------------------------------------------------------- #
DATASET_NAME = "Fashion-MNIST"
DATASET_SOURCE = "https://github.com/zalandoresearch/fashion-mnist"
DATASET_LICENSE = "MIT License (Zalando SE)"

NUM_CLASSES = 10
IMAGE_HEIGHT = 28
IMAGE_WIDTH = 28
CHANNELS = 1
PIXEL_MIN = 0
PIXEL_MAX = 255
PIXEL_SCALE = 255.0

OFFICIAL_TRAIN_SIZE = 60_000
OFFICIAL_TEST_SIZE = 10_000

CLASS_NAMES: tuple[str, ...] = (
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
)

# Experimental subset and preprocessing defaults
RANDOM_SEED = 42
TRAIN_SAMPLES_PER_CLASS = 1_000
TEST_SAMPLES_PER_CLASS = 200
TRAIN_TOTAL = TRAIN_SAMPLES_PER_CLASS * NUM_CLASSES
TEST_TOTAL = TEST_SAMPLES_PER_CLASS * NUM_CLASSES

# Keys stored in the processed .npz archive
ARRAY_KEYS: tuple[str, ...] = ("x_train", "y_train", "x_test", "y_test")


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
def load_experiment_config(path: Path | str = CONFIG_PATH) -> dict[str, Any]:
    """Load the experiment configuration JSON file.

    Args:
        path: Location of the configuration file.

    Returns:
        The parsed configuration dictionary.

    Raises:
        FileNotFoundError: If the configuration file does not exist.
        ValueError: If the file is not a JSON object.
    """
    config_path = Path(path)
    if not config_path.is_file():
        raise FileNotFoundError(f"Experiment configuration not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as handle:
        config = json.load(handle)

    if not isinstance(config, dict):
        raise ValueError(f"Experiment configuration must be a JSON object: {config_path}")
    return config


# --------------------------------------------------------------------------- #
# Validation
# --------------------------------------------------------------------------- #
def class_distribution(labels: np.ndarray) -> dict[int, int]:
    """Count how many samples belong to each class.

    Args:
        labels: Integer class IDs.

    Returns:
        Mapping of class ID to sample count, including classes with zero samples.
    """
    counts = Counter(int(label) for label in np.asarray(labels).ravel())
    return {class_id: counts.get(class_id, 0) for class_id in range(NUM_CLASSES)}


def validate_dataset(
    images: np.ndarray,
    labels: np.ndarray,
    *,
    samples_per_class: int,
    split_name: str,
) -> None:
    """Check shapes, value ranges and class balance of one dataset split.

    Args:
        images: Image array of shape ``(n, 28, 28, 1)``.
        labels: Integer class IDs of shape ``(n,)``.
        samples_per_class: Required number of examples per class.
        split_name: Human-readable split name used in error messages.

    Raises:
        ValueError: If any expectation about the split is violated.
    """
    images = np.asarray(images)
    labels = np.asarray(labels)
    expected_total = samples_per_class * NUM_CLASSES

    if images.ndim != 4:
        raise ValueError(
            f"{split_name}: images must be 4-D (n, h, w, c), got shape {images.shape}"
        )
    if images.shape[1:] != (IMAGE_HEIGHT, IMAGE_WIDTH, CHANNELS):
        raise ValueError(
            f"{split_name}: expected image shape "
            f"({IMAGE_HEIGHT}, {IMAGE_WIDTH}, {CHANNELS}), got {images.shape[1:]}"
        )
    if images.shape[0] != expected_total:
        raise ValueError(
            f"{split_name}: expected {expected_total} images, got {images.shape[0]}"
        )
    if labels.shape != (expected_total,):
        raise ValueError(
            f"{split_name}: expected labels of shape ({expected_total},), got {labels.shape}"
        )
    if not np.issubdtype(images.dtype, np.floating):
        raise ValueError(f"{split_name}: images must be floating point, got {images.dtype}")
    if not np.issubdtype(labels.dtype, np.integer):
        raise ValueError(f"{split_name}: labels must be integers, got {labels.dtype}")
    if images.min() < 0.0 or images.max() > 1.0:
        raise ValueError(
            f"{split_name}: pixel values must lie in [0, 1], "
            f"got [{images.min()}, {images.max()}]"
        )
    if labels.min() < 0 or labels.max() >= NUM_CLASSES:
        raise ValueError(
            f"{split_name}: labels must be class IDs in [0, {NUM_CLASSES - 1}], "
            f"got [{labels.min()}, {labels.max()}]"
        )

    distribution = class_distribution(labels)
    unbalanced = {
        class_id: count
        for class_id, count in distribution.items()
        if count != samples_per_class
    }
    if unbalanced:
        raise ValueError(
            f"{split_name}: every class must contain exactly {samples_per_class} "
            f"examples, violations: {unbalanced}"
        )


# --------------------------------------------------------------------------- #
# Persistence
# --------------------------------------------------------------------------- #
def save_dataset(arrays: Mapping[str, np.ndarray], path: Path | str = DATASET_PATH) -> Path:
    """Save the dataset arrays to a compressed NumPy archive.

    Args:
        arrays: Mapping containing exactly the keys in :data:`ARRAY_KEYS`.
        path: Destination ``.npz`` file; parent directories are created.

    Returns:
        The path that was written.

    Raises:
        ValueError: If required arrays are missing.
    """
    missing = [key for key in ARRAY_KEYS if key not in arrays]
    if missing:
        raise ValueError(f"Cannot save dataset, missing arrays: {missing}")

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, **{key: arrays[key] for key in ARRAY_KEYS})
    return output_path


def load_dataset(path: Path | str = DATASET_PATH) -> dict[str, np.ndarray]:
    """Load the dataset arrays from a NumPy archive.

    Args:
        path: Location of the ``.npz`` file.

    Returns:
        Dictionary with the arrays ``x_train``, ``y_train``, ``x_test``, ``y_test``.

    Raises:
        FileNotFoundError: If the archive does not exist.
        ValueError: If the archive does not contain the expected arrays.
    """
    dataset_path = Path(path)
    if not dataset_path.is_file():
        raise FileNotFoundError(
            f"Processed dataset not found: {dataset_path}. "
            "Run `python -m cnn_scratch.data.prepare_data` first."
        )

    with np.load(dataset_path) as archive:
        missing = [key for key in ARRAY_KEYS if key not in archive.files]
        if missing:
            raise ValueError(f"Dataset archive {dataset_path} is missing arrays: {missing}")
        return {key: archive[key] for key in ARRAY_KEYS}


def save_metadata(metadata: Mapping[str, Any], path: Path | str = METADATA_PATH) -> Path:
    """Write dataset metadata as indented JSON.

    Args:
        metadata: JSON-serialisable metadata dictionary.
        path: Destination file; parent directories are created.

    Returns:
        The path that was written.
    """
    metadata_path = Path(path)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    with metadata_path.open("w", encoding="utf-8") as handle:
        json.dump(dict(metadata), handle, indent=2)
        handle.write("\n")
    return metadata_path


def load_metadata(path: Path | str = METADATA_PATH) -> dict[str, Any]:
    """Read the dataset metadata JSON file.

    Args:
        path: Location of the metadata file.

    Returns:
        The parsed metadata dictionary.

    Raises:
        FileNotFoundError: If the metadata file does not exist.
    """
    metadata_path = Path(path)
    if not metadata_path.is_file():
        raise FileNotFoundError(
            f"Dataset metadata not found: {metadata_path}. "
            "Run `python -m cnn_scratch.data.prepare_data` first."
        )

    with metadata_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)
