"""Prepare the balanced Fashion-MNIST subsets used by both CNN implementations.

Pipeline
--------
1. Download/load the official Fashion-MNIST splits (Keras loader, acquisition only).
2. Validate the official shapes and value ranges.
3. Draw a deterministic, class-balanced training and test subset (seed 42).
4. Normalise to float32 in [0, 1] and add the channel dimension.
5. Validate both subsets, then save the arrays and the metadata file.
6. Save representative sample and class-distribution figures.

The official training and testing splits are never mixed: training samples come
only from the 60,000 training images, test samples only from the 10,000 test
images. TensorFlow/Keras is used here *solely* to download the data.

Run from the repository root:

    python -m cnn_scratch.data.prepare_data
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .dataset_utils import (
    CHANNELS,
    CLASS_NAMES,
    CONFIG_PATH,
    DATASET_NAME,
    DATASET_PATH,
    DATASET_SOURCE,
    FIGURES_DIR,
    IMAGE_HEIGHT,
    IMAGE_WIDTH,
    METADATA_PATH,
    NUM_CLASSES,
    PIXEL_MAX,
    PIXEL_MIN,
    PIXEL_SCALE,
    PROJECT_ROOT,
    RANDOM_SEED,
    TEST_SAMPLES_PER_CLASS,
    TRAIN_SAMPLES_PER_CLASS,
    class_distribution,
    load_experiment_config,
    save_dataset,
    save_metadata,
    validate_dataset,
)
from ..utils.visualization import plot_class_distribution, visualize_samples


def load_fashion_mnist() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load the official Fashion-MNIST training and test splits.

    Keras is imported lazily so that the rest of the pipeline (and the scratch
    implementation) works without importing TensorFlow.

    Returns:
        Tuple ``(x_train, y_train, x_test, y_test)`` as provided by the loader:
        training images ``(60000, 28, 28)`` unsigned bytes, test images
        ``(10000, 28, 28)``, labels as integer class IDs.

    Raises:
        ValueError: If the downloaded splits do not match the official shapes.
    """
    # Local import: dataset acquisition is the only permitted Keras usage here.
    from keras.datasets import fashion_mnist

    (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

    for split_name, images, labels, expected_size in (
        ("official training", x_train, y_train, 60_000),
        ("official test", x_test, y_test, 10_000),
    ):
        if images.shape != (expected_size, IMAGE_HEIGHT, IMAGE_WIDTH):
            raise ValueError(
                f"{split_name} split has unexpected shape {images.shape}; "
                f"expected {(expected_size, IMAGE_HEIGHT, IMAGE_WIDTH)}"
            )
        if labels.shape != (expected_size,):
            raise ValueError(
                f"{split_name} labels have unexpected shape {labels.shape}; "
                f"expected {(expected_size,)}"
            )
        if images.min() < PIXEL_MIN or images.max() > PIXEL_MAX:
            raise ValueError(
                f"{split_name} pixel values outside [{PIXEL_MIN}, {PIXEL_MAX}]: "
                f"[{images.min()}, {images.max()}]"
            )

    return x_train, y_train, x_test, y_test


def select_balanced_subset(
    images: np.ndarray,
    labels: np.ndarray,
    *,
    samples_per_class: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Draw an equal number of samples per class using a seeded generator.

    Sampling is without replacement, so a subset never repeats an image. The
    selected indices are shuffled so that classes are not stored contiguously.

    Args:
        images: Source images of shape ``(n, 28, 28)``.
        labels: Source class IDs of shape ``(n,)``.
        samples_per_class: Number of images to draw per class.
        seed: Seed for ``numpy.random.default_rng``.

    Returns:
        Tuple of subset images and labels, both ordered consistently.

    Raises:
        ValueError: If a class has fewer samples than requested.
    """
    images = np.asarray(images)
    labels = np.asarray(labels)

    if samples_per_class <= 0:
        raise ValueError(f"samples_per_class must be positive, got {samples_per_class}")
    if images.shape[0] != labels.shape[0]:
        raise ValueError(
            f"images and labels disagree on sample count: {images.shape[0]} vs {labels.shape[0]}"
        )

    rng = np.random.default_rng(seed)
    selected_indices: list[np.ndarray] = []

    for class_id in range(NUM_CLASSES):
        class_indices = np.flatnonzero(labels == class_id)
        if class_indices.size < samples_per_class:
            raise ValueError(
                f"Class {class_id} ({CLASS_NAMES[class_id]}) has only "
                f"{class_indices.size} samples; {samples_per_class} requested"
            )
        selected_indices.append(
            rng.choice(class_indices, size=samples_per_class, replace=False)
        )

    indices = np.concatenate(selected_indices)
    indices = rng.permutation(indices)  # deterministic shuffle of the selection
    return images[indices], labels[indices]


def normalize_images(images: np.ndarray) -> np.ndarray:
    """Scale pixel values to [0, 1], cast to float32 and append the channel axis.

    Args:
        images: Images of shape ``(n, 28, 28)`` with values in [0, 255].

    Returns:
        Array of shape ``(n, 28, 28, 1)`` with dtype ``float32``.

    Raises:
        ValueError: If pixel values fall outside [0, 255].
    """
    images = np.asarray(images)
    if images.min() < PIXEL_MIN or images.max() > PIXEL_MAX:
        raise ValueError(
            f"Cannot normalise images with values outside [{PIXEL_MIN}, {PIXEL_MAX}]: "
            f"[{images.min()}, {images.max()}]"
        )

    normalized = images.astype(np.float32) / PIXEL_SCALE
    return normalized[..., np.newaxis]


def _relative_posix(path: Path) -> str:
    """Return a repository-relative POSIX path so metadata stays portable."""
    try:
        return path.resolve().relative_to(PROJECT_ROOT).as_posix()
    except ValueError:  # output written outside the repository
        return path.as_posix()


def _split_metadata(
    images: np.ndarray, labels: np.ndarray, official_size: int
) -> dict[str, Any]:
    """Describe one prepared split for the metadata file."""
    distribution = class_distribution(labels)
    return {
        "shape": list(images.shape),
        "dtype": str(images.dtype),
        "label_dtype": str(labels.dtype),
        "num_samples": int(images.shape[0]),
        "official_split_size": official_size,
        "class_distribution": {str(k): v for k, v in distribution.items()},
        "class_distribution_named": {
            CLASS_NAMES[int(k)]: v for k, v in distribution.items()
        },
    }


def build_metadata(
    *,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    seed: int,
    train_samples_per_class: int,
    test_samples_per_class: int,
    output_file: Path,
) -> dict[str, Any]:
    """Assemble the dataset metadata record.

    Args:
        x_train: Prepared training images.
        y_train: Training labels.
        x_test: Prepared test images.
        y_test: Test labels.
        seed: Random seed used for subset selection.
        train_samples_per_class: Requested training samples per class.
        test_samples_per_class: Requested test samples per class.
        output_file: Path of the saved dataset archive.

    Returns:
        JSON-serialisable metadata dictionary.
    """
    return {
        "dataset": DATASET_NAME,
        "source": DATASET_SOURCE,
        "source_loader": "keras.datasets.fashion_mnist.load_data (acquisition only)",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "random_seed": seed,
        "num_classes": NUM_CLASSES,
        "class_labels": {str(i): name for i, name in enumerate(CLASS_NAMES)},
        "image_shape": [IMAGE_HEIGHT, IMAGE_WIDTH, CHANNELS],
        "subset_sizes": {
            "train_samples_per_class": train_samples_per_class,
            "test_samples_per_class": test_samples_per_class,
            "train_samples_total": int(x_train.shape[0]),
            "test_samples_total": int(x_test.shape[0]),
        },
        "preprocessing": {
            "dtype": "float32",
            "normalization": "divide_by_255",
            "source_pixel_range": [PIXEL_MIN, PIXEL_MAX],
            "output_pixel_range": [0.0, 1.0],
            "channel_dimension": "added as trailing axis (n, 28, 28, 1)",
            "labels": "integer class IDs, no one-hot encoding",
            "sampling": (
                "balanced per-class sampling without replacement from the official "
                "split, then a seeded permutation of the selected indices"
            ),
            "train_test_mixing": "none; official training and test splits stay separate",
        },
        "splits": {
            "train": _split_metadata(x_train, y_train, 60_000),
            "test": _split_metadata(x_test, y_test, 10_000),
        },
        "files": {
            "dataset": _relative_posix(Path(output_file)),
            "arrays": ["x_train", "y_train", "x_test", "y_test"],
        },
        "usage_note": (
            "The test split is held out for final evaluation only; it must not be "
            "used for training or hyperparameter tuning."
        ),
    }


def prepare_dataset(
    config: Mapping[str, Any] | None = None,
    *,
    output_path: Path | str = DATASET_PATH,
    metadata_path: Path | str = METADATA_PATH,
    save_figure: bool = True,
    figures_dir: Path | str = FIGURES_DIR,
) -> dict[str, Any]:
    """Run the full preparation pipeline and persist the artefacts.

    Args:
        config: Experiment configuration; defaults to ``configs/experiment_config.json``.
        output_path: Destination ``.npz`` archive.
        metadata_path: Destination metadata JSON file.
        save_figure: Whether to write the dataset visualisations.
        figures_dir: Directory for generated figures.

    Returns:
        Dictionary with ``arrays``, ``metadata``, ``output_path`` and
        ``metadata_path``.
    """
    settings = dict(config) if config is not None else load_experiment_config(CONFIG_PATH)

    seed = int(settings.get("random_seed", RANDOM_SEED))
    train_per_class = int(settings.get("train_samples_per_class", TRAIN_SAMPLES_PER_CLASS))
    test_per_class = int(settings.get("test_samples_per_class", TEST_SAMPLES_PER_CLASS))

    x_train_raw, y_train_raw, x_test_raw, y_test_raw = load_fashion_mnist()

    x_train_sub, y_train_sub = select_balanced_subset(
        x_train_raw, y_train_raw, samples_per_class=train_per_class, seed=seed
    )
    # Offset the test seed so the two splits are not drawn from identical indices.
    x_test_sub, y_test_sub = select_balanced_subset(
        x_test_raw, y_test_raw, samples_per_class=test_per_class, seed=seed + 1
    )

    x_train = normalize_images(x_train_sub)
    x_test = normalize_images(x_test_sub)

    validate_dataset(
        x_train, y_train_sub, samples_per_class=train_per_class, split_name="training"
    )
    validate_dataset(
        x_test, y_test_sub, samples_per_class=test_per_class, split_name="test"
    )

    arrays = {
        "x_train": x_train,
        "y_train": y_train_sub,
        "x_test": x_test,
        "y_test": y_test_sub,
    }

    saved_dataset = save_dataset(arrays, output_path)
    metadata = build_metadata(
        x_train=x_train,
        y_train=y_train_sub,
        x_test=x_test,
        y_test=y_test_sub,
        seed=seed,
        train_samples_per_class=train_per_class,
        test_samples_per_class=test_per_class,
        output_file=Path(saved_dataset),
    )
    saved_metadata = save_metadata(metadata, metadata_path)

    if save_figure:
        visualize_samples(
            x_train,
            y_train_sub,
            Path(figures_dir) / "dataset_samples.png",
            samples_per_class=min(10, train_per_class),
        )
        plot_class_distribution(
            [("train", y_train_sub), ("test", y_test_sub)],
            Path(figures_dir) / "class_distribution.png",
        )

    return {
        "arrays": arrays,
        "metadata": metadata,
        "output_path": Path(saved_dataset),
        "metadata_path": Path(saved_metadata),
    }


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments for the preparation script."""
    parser = argparse.ArgumentParser(
        description="Prepare the balanced Fashion-MNIST subsets used in this project."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=CONFIG_PATH,
        help=f"Experiment configuration file (default: {CONFIG_PATH.name})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DATASET_PATH,
        help="Destination .npz archive (default: data/processed/fashion_mnist_subset.npz)",
    )
    parser.add_argument(
        "--metadata",
        type=Path,
        default=METADATA_PATH,
        help="Destination metadata JSON file",
    )
    parser.add_argument(
        "--no-figure",
        action="store_true",
        help="Skip writing the dataset visualisations",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point for ``python -m cnn_scratch.data.prepare_data``."""
    args = _parse_args(argv)
    result = prepare_dataset(
        load_experiment_config(args.config),
        output_path=args.output,
        metadata_path=args.metadata,
        save_figure=not args.no_figure,
    )

    arrays = result["arrays"]
    metadata = result["metadata"]
    print("Fashion-MNIST subsets prepared")
    print(f"  train: {arrays['x_train'].shape} {arrays['x_train'].dtype}")
    print(f"  test : {arrays['x_test'].shape} {arrays['x_test'].dtype}")
    print(f"  train class counts: {metadata['splits']['train']['class_distribution']}")
    print(f"  test  class counts: {metadata['splits']['test']['class_distribution']}")
    print(f"  dataset : {result['output_path']}")
    print(f"  metadata: {result['metadata_path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
