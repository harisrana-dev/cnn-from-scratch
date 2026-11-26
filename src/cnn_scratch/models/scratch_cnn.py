"""Scratch CNN model assembled from the frozen layer specification.

This module ties together the layers in ``layers.py`` exactly as the frozen
architecture declares::

    Conv2D(8, 3x3, stride 1, valid) -> ReLU -> MaxPool2D(2x2, stride 2) ->
    Flatten -> Dense(10) -> Softmax

Forward propagation runs every layer in order and returns the final
probabilities.  Backpropagation runs the layers in reverse order and returns
every gradient, all the way to the input.

This module deliberately does **not** normalise gradients or update
parameters: the SDG loop (Phase 4) owns the learning-rate schedule and the
parameter updates.
"""

from __future__ import annotations

from typing import Any, Iterable, Sequence

import numpy as np

from .layers import (
    CategoricalCrossEntropy,
    Conv2D,
    Dense,
    Flatten,
    MaxPool2D,
    ReLU,
    StableSoftmax,
)

#: Order of layers in the frozen architecture.
LAYER_ORDER: tuple[str, ...] = ("conv2d", "relu", "maxpool2d", "flatten", "dense", "softmax")


class ScratchCNN:
    """Forward + backward pass for the frozen NumPy architecture."""

    def __init__(self, layers: Iterable[dict[str, Any]], *, seed: int = 42) -> None:
        """Assemble the layers declared in ``configs/experiment_config.json``.

        Args:
            layers: The architecture layer list, in forward order.
            seed: Random seed used for reproducible weight initialisation.

        Raises:
            ValueError: If a layer type is not yet implemented.
        """
        self.layers: list[dict[str, Any]] = []
        self.conv: Conv2D | None = None
        self.relu: ReLU = ReLU()
        self.pool: MaxPool2D | None = None
        self.flatten: Flatten = Flatten()
        self.dense: Dense | None = None
        self.softmax: StableSoftmax = StableSoftmax()
        self.loss: CategoricalCrossEntropy = CategoricalCrossEntropy()

        for entry in layers:
            kind = entry.get("type")
            if kind == "conv2d":
                self.conv = Conv2D(
                    (entry["kernel_size"], entry["kernel_size"],
                     entry["channels"], entry["filters"]),
                    seed=seed,
                )
            elif kind == "relu":
                self.relu = ReLU()
            elif kind == "maxpool2d":
                self.pool = MaxPool2D(
                    pool_size=entry["pool_size"],
                    stride=entry.get("stride", entry["pool_size"]),
                )
            elif kind == "flatten":
                self.flatten = Flatten()
            elif kind == "dense":
                self.dense = Dense(entry["inputs"], entry["units"], seed=seed)
            elif kind == "softmax":
                self.softmax = StableSoftmax()
            else:
                raise ValueError(f"unsupported layer type in spec: {kind!r}")

        if not all([self.conv, self.dense]):
            raise ValueError("the scratch model needs a Conv2D and a Dense layer")

    # -- forward ------------------------------------------------------------ #

    def forward(self, x: np.ndarray, y: np.ndarray | None = None) -> np.ndarray | tuple[np.ndarray, float]:
        """Push ``x`` through the whole network.

        Args:
            x: ``(B, 28, 28, 1)`` NHWC input.

        Returns:
            The class probabilities ``(B, 10)``, or ``(probabilities, loss)``
            when a target ``y`` is supplied.
        """
        out = self.conv.forward(x)
        out = self.relu.forward(out)
        out = self.pool.forward(out)
        out = self.flatten.forward(out)
        logits = self.dense.forward(out)
        probs = self.softmax.forward(logits)
        if y is None:
            return probs
        return probs, self.loss.forward(probs, y)

    # -- backward ----------------------------------------------------------- #

    def backward(self, dy: np.ndarray, y: np.ndarray) -> dict[str, np.ndarray]:
        """Backpropagate and return every gradient.

        Args:
            dy: Upstream gradient w.r.t. the probabilities, already carrying
                the batch-averaging factor (``(P - Y) / B``).
            y: One-hot targets ``(B, 10)``.

        Returns:
            Dict with keys ``"x"``, ``"W1"``, ``"b1"``, ``"W2"``, ``"b2"``.
        """
        # Loss gradient already applied: softmax.backward returns (P - Y) / B
        # once, which is exactly the upstream ``dy`` this method expects.
        dlogits = self.softmax.backward(dy if dy is not None else self.loss.backward())

        dW2, db2 = None, None
        out = self.dense.backward(dlogits)
        if isinstance(out, tuple):
            dx, dW2, db2 = out
        else:  # single-sample dense (no batch)
            dx, dW2, db2 = out, _zero_like(self.dense.W), _zero_like(self.dense.b)

        out = self.flatten.backward(dx)
        out = self.pool.backward(out)
        out = self.relu.backward(out)
        dx, dW1, db1 = self.conv.backward(out)

        return {"x": dx, "W1": dW1, "b1": db1, "W2": dW2, "b2": db2}

    # -- helpers ------------------------------------------------------------ #

    def parameters(self) -> list[np.ndarray]:
        """All trainable parameters, in a deterministic order."""
        params: list[np.ndarray] = []
        if self.conv is not None:
            params.append(self.conv.W)
            params.append(self.conv.b)
        if self.dense is not None:
            params.append(self.dense.W)
            params.append(self.dense.b)
        return params


def _zero_like(arr: np.ndarray) -> np.ndarray:
    return np.zeros_like(arr, dtype=np.float64)
