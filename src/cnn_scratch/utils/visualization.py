"""Figure helpers used to inspect the prepared dataset.

The non-interactive Agg backend is selected explicitly so figures render the
same way in a terminal, in VS Code and on a headless machine.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402  (backend must be set first)
import numpy as np  # noqa: E402

from ..data.dataset_utils import CLASS_NAMES, NUM_CLASSES  # noqa: E402


def _ensure_parent(output_path: Path) -> None:
    """Create the parent directory of a figure if needed."""
    output_path.parent.mkdir(parents=True, exist_ok=True)


def visualize_samples(
    images: np.ndarray,
    labels: np.ndarray,
    output_path: Path | str,
    *,
    samples_per_class: int = 10,
    title: str = "Fashion-MNIST: representative samples per class",
) -> Path:
    """Save a grid figure with one row per class and ``samples_per_class`` columns.

    Args:
        images: Images of shape ``(n, 28, 28, 1)`` with values in [0, 1].
        labels: Integer class IDs of shape ``(n,)``.
        output_path: Destination image file (``.png``).
        samples_per_class: Number of example images to draw per class.
        title: Figure title.

    Returns:
        The path that was written.

    Raises:
        ValueError: If the array shapes are inconsistent or no samples exist.
    """
    images = np.asarray(images)
    labels = np.asarray(labels)
    if images.shape[0] != labels.shape[0]:
        raise ValueError(
            f"images and labels disagree on sample count: {images.shape[0]} vs {labels.shape[0]}"
        )
    if images.shape[0] == 0:
        raise ValueError("Cannot visualise an empty dataset")

    figure, axes = plt.subplots(
        nrows=NUM_CLASSES,
        ncols=samples_per_class,
        figsize=(samples_per_class * 1.1, NUM_CLASSES * 1.1),
    )
    # np.ndarray indexing is used below, so normalise the single-cell case.
    axes = np.atleast_2d(axes)

    for class_id in range(NUM_CLASSES):
        class_indices = np.flatnonzero(labels == class_id)
        for column in range(samples_per_class):
            axis = axes[class_id, column]
            axis.axis("off")
            if column < class_indices.size:
                image = images[class_indices[column]]
                axis.imshow(image.squeeze(), cmap="gray", vmin=0.0, vmax=1.0)
            if column == 0:
                axis.set_ylabel(
                    f"{class_id} {CLASS_NAMES[class_id]}",
                    rotation=0,
                    ha="right",
                    va="center",
                    fontsize=8,
                )

    figure.suptitle(title, fontsize=12)
    figure.tight_layout(rect=(0.02, 0.0, 1.0, 0.97))

    destination = Path(output_path)
    _ensure_parent(destination)
    figure.savefig(destination, dpi=150, bbox_inches="tight")
    plt.close(figure)
    return destination


def plot_class_distribution(
    label_sets: Sequence[tuple[str, np.ndarray]],
    output_path: Path | str,
    *,
    title: str = "Fashion-MNIST subset class distribution",
) -> Path:
    """Save a grouped bar chart of per-class sample counts.

    Args:
        label_sets: Sequence of ``(split_name, labels)`` pairs.
        output_path: Destination image file (``.png``).
        title: Figure title.

    Returns:
        The path that was written.

    Raises:
        ValueError: If ``label_sets`` is empty.
    """
    if not label_sets:
        raise ValueError("At least one label set is required")

    figure, axis = plt.subplots(figsize=(10, 5))
    positions = np.arange(NUM_CLASSES)
    bar_width = 0.8 / len(label_sets)

    for offset, (split_name, labels) in enumerate(label_sets):
        labels = np.asarray(labels)
        counts = np.bincount(labels, minlength=NUM_CLASSES)
        axis.bar(
            positions + offset * bar_width,
            counts,
            width=bar_width,
            label=f"{split_name} ({labels.size} total)",
        )

    axis.set_xticks(positions)
    axis.set_xticklabels(
        [f"{i}\n{name}" for i, name in enumerate(CLASS_NAMES)],
        fontsize=8,
        rotation=45,
        ha="right",
    )
    axis.set_ylabel("Samples")
    axis.set_title(title)
    axis.legend()
    figure.tight_layout()

    destination = Path(output_path)
    _ensure_parent(destination)
    figure.savefig(destination, dpi=150, bbox_inches="tight")
    plt.close(figure)
    return destination
