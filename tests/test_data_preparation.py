"""Tests for the Fashion-MNIST preparation pipeline.

The shape, dtype, normalisation and balance tests run against the real
processed dataset. If ``data/processed/fashion_mnist_subset.npz`` is missing,
the session fixture rebuilds it with the project's own pipeline, so the tests
double as an end-to-end check of ``prepare_data``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from cnn_scratch.data.dataset_utils import (
    ARRAY_KEYS,
    CHANNELS,
    IMAGE_HEIGHT,
    IMAGE_WIDTH,
    METADATA_PATH,
    NUM_CLASSES,
    RANDOM_SEED,
    TEST_SAMPLES_PER_CLASS,
    TRAIN_SAMPLES_PER_CLASS,
    class_distribution,
    load_dataset,
    load_metadata,
    save_dataset,
    validate_dataset,
)
from cnn_scratch.data.prepare_data import (
    normalize_images,
    prepare_dataset,
    select_balanced_subset,
)

TRAIN_TOTAL = TRAIN_SAMPLES_PER_CLASS * NUM_CLASSES
TEST_TOTAL = TEST_SAMPLES_PER_CLASS * NUM_CLASSES


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="session")
def dataset() -> tuple[dict[str, np.ndarray], dict]:
    """Return the prepared arrays and metadata, building them if necessary."""
    from cnn_scratch.data.dataset_utils import DATASET_PATH

    if not DATASET_PATH.is_file() or not METADATA_PATH.is_file():
        prepare_dataset(save_figure=False)
    return load_dataset(DATASET_PATH), load_metadata(METADATA_PATH)


@pytest.fixture()
def synthetic_split() -> tuple[np.ndarray, np.ndarray]:
    """Small synthetic split with an unbalanced-class shape for unit tests."""
    rng = np.random.default_rng(0)
    images = rng.integers(0, 256, size=(NUM_CLASSES * 20, IMAGE_HEIGHT, IMAGE_WIDTH))
    labels = np.repeat(np.arange(NUM_CLASSES, dtype=np.uint8), 20)
    return images, labels


# --------------------------------------------------------------------------- #
# Prepared dataset: shapes, dtypes, ranges
# --------------------------------------------------------------------------- #
def test_training_shape(dataset):
    arrays, _ = dataset
    assert arrays["x_train"].shape == (TRAIN_TOTAL, IMAGE_HEIGHT, IMAGE_WIDTH, CHANNELS)
    assert arrays["y_train"].shape == (TRAIN_TOTAL,)


def test_testing_shape(dataset):
    arrays, _ = dataset
    assert arrays["x_test"].shape == (TEST_TOTAL, IMAGE_HEIGHT, IMAGE_WIDTH, CHANNELS)
    assert arrays["y_test"].shape == (TEST_TOTAL,)


def test_image_dimensions_and_channel_axis(dataset):
    arrays, _ = dataset
    for split in ("train", "test"):
        images = arrays[f"x_{split}"]
        assert images.ndim == 4
        assert images.shape[1] == IMAGE_HEIGHT
        assert images.shape[2] == IMAGE_WIDTH
        assert images.shape[3] == CHANNELS


def test_images_are_float32(dataset):
    arrays, _ = dataset
    assert arrays["x_train"].dtype == np.float32
    assert arrays["x_test"].dtype == np.float32


def test_pixel_values_within_unit_interval(dataset):
    arrays, _ = dataset
    for split in ("train", "test"):
        images = arrays[f"x_{split}"]
        assert float(images.min()) >= 0.0
        assert float(images.max()) <= 1.0


def test_labels_are_integer_class_ids(dataset):
    arrays, _ = dataset
    for split in ("train", "test"):
        labels = arrays[f"y_{split}"]
        assert np.issubdtype(labels.dtype, np.integer)
        assert set(np.unique(labels).tolist()) == set(range(NUM_CLASSES))


def test_training_class_balance(dataset):
    arrays, _ = dataset
    distribution = class_distribution(arrays["y_train"])
    assert sum(distribution.values()) == TRAIN_TOTAL
    assert all(count == TRAIN_SAMPLES_PER_CLASS for count in distribution.values())


def test_testing_class_balance(dataset):
    arrays, _ = dataset
    distribution = class_distribution(arrays["y_test"])
    assert sum(distribution.values()) == TEST_TOTAL
    assert all(count == TEST_SAMPLES_PER_CLASS for count in distribution.values())


def test_validate_dataset_accepts_prepared_splits(dataset):
    arrays, _ = dataset
    validate_dataset(
        arrays["x_train"], arrays["y_train"],
        samples_per_class=TRAIN_SAMPLES_PER_CLASS, split_name="training",
    )
    validate_dataset(
        arrays["x_test"], arrays["y_test"],
        samples_per_class=TEST_SAMPLES_PER_CLASS, split_name="test",
    )


# --------------------------------------------------------------------------- #
# Metadata
# --------------------------------------------------------------------------- #
def test_metadata_records_specification(dataset):
    _, metadata = dataset
    assert metadata["dataset"] == "Fashion-MNIST"
    assert metadata["random_seed"] == RANDOM_SEED
    assert metadata["num_classes"] == NUM_CLASSES
    assert metadata["image_shape"] == [IMAGE_HEIGHT, IMAGE_WIDTH, CHANNELS]
    assert "fashion-mnist" in metadata["source"].lower()
    assert metadata["preprocessing"]["normalization"] == "divide_by_255"
    assert metadata["preprocessing"]["dtype"] == "float32"
    assert len(metadata["class_labels"]) == NUM_CLASSES


def test_metadata_matches_saved_arrays(dataset):
    arrays, metadata = dataset
    assert metadata["subset_sizes"]["train_samples_total"] == arrays["y_train"].size
    assert metadata["subset_sizes"]["test_samples_total"] == arrays["y_test"].size
    assert metadata["subset_sizes"]["train_samples_per_class"] == TRAIN_SAMPLES_PER_CLASS
    assert metadata["subset_sizes"]["test_samples_per_class"] == TEST_SAMPLES_PER_CLASS
    assert metadata["splits"]["train"]["shape"] == list(arrays["x_train"].shape)
    assert metadata["splits"]["test"]["shape"] == list(arrays["x_test"].shape)
    assert {
        int(key): value
        for key, value in metadata["splits"]["train"]["class_distribution"].items()
    } == class_distribution(arrays["y_train"])


# --------------------------------------------------------------------------- #
# Determinism and unit behaviour
# --------------------------------------------------------------------------- #
def test_sampling_is_deterministic(synthetic_split):
    images, labels = synthetic_split
    first_images, first_labels = select_balanced_subset(
        images, labels, samples_per_class=5, seed=RANDOM_SEED
    )
    second_images, second_labels = select_balanced_subset(
        images, labels, samples_per_class=5, seed=RANDOM_SEED
    )
    np.testing.assert_array_equal(first_images, second_images)
    np.testing.assert_array_equal(first_labels, second_labels)


def test_repeated_subsets_draw_identical_indices(dataset):
    """Re-running the pipeline with the same seed must reproduce the split."""
    _, metadata = dataset
    again = prepare_dataset(save_figure=False)
    try:
        arrays, _ = dataset
        np.testing.assert_array_equal(again["arrays"]["y_train"], arrays["y_train"])
        np.testing.assert_array_equal(again["arrays"]["x_train"], arrays["x_train"])
        np.testing.assert_array_equal(again["arrays"]["x_test"], arrays["x_test"])
        assert metadata["random_seed"] == again["metadata"]["random_seed"]
    finally:
        # Restore the canonical artefacts and metadata for later tests.
        from cnn_scratch.data.dataset_utils import DATASET_PATH, save_metadata

        save_dataset(again["arrays"], DATASET_PATH)
        save_metadata(metadata, METADATA_PATH)


def test_different_seed_changes_selection(synthetic_split):
    images, labels = synthetic_split
    _, first_labels = select_balanced_subset(
        images, labels, samples_per_class=5, seed=RANDOM_SEED
    )
    _, second_labels = select_balanced_subset(
        images, labels, samples_per_class=5, seed=RANDOM_SEED + 1
    )
    assert class_distribution(first_labels) == class_distribution(second_labels)
    assert not np.array_equal(first_labels, second_labels)


def test_select_balanced_subset_is_balanced(synthetic_split):
    images, labels = synthetic_split
    subset_images, subset_labels = select_balanced_subset(
        images, labels, samples_per_class=7, seed=1
    )
    assert subset_images.shape == (NUM_CLASSES * 7, IMAGE_HEIGHT, IMAGE_WIDTH)
    assert class_distribution(subset_labels) == {
        class_id: 7 for class_id in range(NUM_CLASSES)
    }


def test_select_balanced_subset_rejects_too_many_samples(synthetic_split):
    images, labels = synthetic_split
    with pytest.raises(ValueError, match="samples"):
        select_balanced_subset(images, labels, samples_per_class=21, seed=1)


def test_normalize_images_scales_and_adds_channel(synthetic_split):
    images, _ = synthetic_split
    normalized = normalize_images(images)
    assert normalized.shape == (*images.shape, CHANNELS)
    assert normalized.dtype == np.float32
    assert float(normalized.min()) >= 0.0
    assert float(normalized.max()) <= 1.0
    np.testing.assert_allclose(normalized[..., 0], images.astype(np.float32) / 255.0)


def test_normalize_images_rejects_out_of_range_values():
    images = np.full((2, IMAGE_HEIGHT, IMAGE_WIDTH), 300, dtype=np.int32)
    with pytest.raises(ValueError, match="outside"):
        normalize_images(images)


def test_validate_dataset_rejects_wrong_image_shape(synthetic_split):
    images, labels = synthetic_split
    normalized = normalize_images(images)
    # Keep the sample count correct so the spatial-shape check is reached.
    wrong_shape = np.zeros(
        (normalized.shape[0], IMAGE_HEIGHT + 4, IMAGE_WIDTH, CHANNELS), dtype=np.float32
    )
    with pytest.raises(ValueError, match="image shape"):
        validate_dataset(
            wrong_shape, labels, samples_per_class=20, split_name="synthetic"
        )


def test_validate_dataset_rejects_wrong_sample_count(synthetic_split):
    images, labels = synthetic_split
    normalized = normalize_images(images)
    with pytest.raises(ValueError, match="expected 200 images"):
        validate_dataset(
            normalized[:-1], labels[:-1], samples_per_class=20, split_name="synthetic"
        )


def test_validate_dataset_rejects_unnormalized_pixels(synthetic_split):
    images, labels = synthetic_split
    unnormalized = images.astype(np.float32)[..., np.newaxis]
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        validate_dataset(
            unnormalized, labels, samples_per_class=20, split_name="synthetic"
        )


# --------------------------------------------------------------------------- #
# Persistence
# --------------------------------------------------------------------------- #
def test_save_and_reload_preserve_arrays(tmp_path: Path, synthetic_split):
    images, labels = synthetic_split
    arrays = {
        "x_train": normalize_images(images),
        "y_train": labels,
        "x_test": normalize_images(images[:40]),
        "y_test": labels[:40],
    }
    destination = tmp_path / "nested" / "subset.npz"
    save_dataset(arrays, destination)

    assert destination.is_file()
    reloaded = load_dataset(destination)
    assert set(reloaded) == set(ARRAY_KEYS)
    for key, value in arrays.items():
        np.testing.assert_array_equal(reloaded[key], value)
        assert reloaded[key].dtype == value.dtype


def test_save_dataset_requires_all_arrays(tmp_path: Path):
    with pytest.raises(ValueError, match="missing"):
        save_dataset({"x_train": np.zeros((1, 28, 28, 1), dtype=np.float32)}, tmp_path / "x.npz")


def test_load_dataset_reports_missing_file(tmp_path: Path):
    with pytest.raises(FileNotFoundError, match="prepare_data"):
        load_dataset(tmp_path / "absent.npz")
