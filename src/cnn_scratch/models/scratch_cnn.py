"""Scratch CNN model assembled from the frozen layer specification.

This module ties together the layers in ``layers.py`` exactly as the frozen
architecture declares::

    Conv2D(8, 3x3, stride 1, valid) -> ReLU -> MaxPool2D(2x2, stride 2) ->
    Flatten -> Dense(10) -> Softmax

Forward propagation runs every layer in order and returns the final
probabilities.  Backpropagation runs the layers in reverse order and returns
every gradient, all the way to the input.

Gradient contract
-----------------
* The mean categorical cross-entropy loss averages over the batch once, at the
  loss level.
* The softmax + cross-entropy logits gradient is the **fused** simplified
  gradient ``G = (P - Y) / B`` (see Section 14 of
  ``docs/cnn_mathematical_foundations.md``).  This is computed exactly once,
  either by the caller passing ``dy = (P - Y) / B`` into :meth:`backward`, or
  by calling ``backward`` with ``dy=None``, in which case this class computes
  the fused gradient from the cached probabilities.
* ``StableSoftmax.backward`` is the **general softmax vector-Jacobian product**
  and must NOT be passed an already-fused logits gradient, because doing so
  applies the softmax derivative a second time.  The fused path in this class
  therefore does not route through :meth:`StableSoftmax.backward`.
* Convolution and dense parameter gradients are batch-averaged exactly once
  (dense: ``A.T @ G`` with ``G`` already carrying the ``1/B``; conv: the
  mean over the batch is folded into the upstream ``dy``).
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
        self._last_logits: np.ndarray | None = None
        self._last_probs: np.ndarray | None = None

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

    # -- forward ------------------------------------------------------------

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
        self._last_logits = logits
        self._last_probs = probs
        if y is None:
            return probs
        return probs, self.loss.forward(probs, y)

    # -- backward -----------------------------------------------------------

    def backward(
        self, dy: np.ndarray | None, y: np.ndarray | None = None
    ) -> dict[str, np.ndarray]:
        """Backpropagate and return every gradient.

        Args:
            dy: Upstream gradient w.r.t. the pre-softmax logits, **already
                carrying the batch-averaging factor ``(P - Y) / B``**.  This
                value already contains the single ``1 / B`` of the mean loss,
                so no layer in this chain divides by ``B`` again.  Pass
                ``None`` to compute it automatically as
                ``(self._last_probs - y) / B`` from the cached probabilities.
            y: One-hot targets ``(B, 10)``.  Required when ``dy is None``.

        Returns:
            Dict with keys ``"x"``, ``"W1"``, ``"b1"``, ``"W2"``, ``"b2"``.
        """
        if dy is None:
            if y is None:
                raise ValueError(
                    "backward(dy=None) requires a one-hot target y to form "
                    "(P - Y) / B"
                )
            if self._last_probs is None:
                raise RuntimeError("backward(dy=None) must follow a forward(x, y) call")
            dy = (self._last_probs - y) / self._last_probs.shape[0]

        # Fused softmax + cross-entropy path: the logits gradient is
        # (P - Y) / B.  We hand it straight to the dense layer; it must never
        # pass through StableSoftmax.backward, which is the general
        # vector-Jacobian product and would be applied a second time.
        dlogits = dy
        out = self.dense.backward(dlogits)
        dx, dW2, db2 = out

        out = self.flatten.backward(dx)
        out = self.pool.backward(out)
        out = self.relu.backward(out)
        dx, dW1, db1 = self.conv.backward(out)

        return {"x": dx, "W1": dW1, "b1": db1, "W2": dW2, "b2": db2}

    # -- helpers ------------------------------------------------------------

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

    @property
    def _probs(self) -> np.ndarray | None:
        """Cached probabilities from the last forward pass."""
        return getattr(self, "_last_probs", None)
