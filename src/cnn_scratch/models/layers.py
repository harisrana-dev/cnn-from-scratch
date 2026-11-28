"""NumPy CNN layers for the frozen architecture.

This module implements the manual forward and backward passes of every layer
the scratch model needs. It is pure NumPy: no TensorFlow, Keras, PyTorch, JAX
or any autograd framework is used anywhere in it.

Conventions
-----------
* The first axis of every tensor is the batch axis; all other axes are the
  per-sample shape and are preserved by every layer (never flattened across
  the batch dimension).
* Convolution is cross-correlation: the kernel is *not* flipped.
  ``Y[b, i, j, f] = sum_{u,v,c} X[b, i+u, j+v, c] * W[u, v, c, f]``
* Convolution weights are stored ``(KH, KW, Cin, Cout)``.
* Dense weights are stored ``(input_features, output_features)``.
* Every layer caches exactly what its backward pass needs, and never mutates a
  cached array in place.
* Parameter gradients are accumulated from the *same* forward-pass parameter
  values. Layers do not update weights; that is the job of the training loop.

All layers accept an arbitrary number of batch-input dimensions (scalar,
vector, tensor, ...) where the leading axis is always interpreted as the
sample/batch axis.
"""


from __future__ import annotations

from typing import Any

import numpy as np

# ---------------------------------------------------------------------------
# Helpers shared by several layers
# ---------------------------------------------------------------------------


def _as_batch(x: np.ndarray) -> np.ndarray:
    """Give the input an explicit leading batch dimension if needed.

    A 1-D, 2-D or 3-D input gets a leading axis prepended; anything with a
    leading batch dimension (2-D, 3-D, ... including (B, ...)) passes through.
    """
    if x.ndim == 0:
        raise ValueError("scalar input is not valid; a batch axis is required")
    if x.ndim == 1:
        return x[np.newaxis, ...]
    return x


def _squeeze_batch(x: np.ndarray) -> np.ndarray:
    """Remove a leading singleton batch axis, or return ``x`` unchanged."""
    if x.ndim >= 2 and x.shape[0] == 1:
        return x[0]
    return x


def _zero(*shape: int) -> np.ndarray:
    """Return a zero array of the given shape (float64 to match float64 checks)."""
    return np.zeros(shape, dtype=np.float64)


def _randn(*shape: int, scale: float = 1e-3) -> np.ndarray:
    """Small deterministic-ish seed for gradient checks (not the model init)."""
    return np.random.default_rng(1234).standard_normal(shape).astype(np.float64) * scale


# ---------------------------------------------------------------------------
# Conv2D
# ---------------------------------------------------------------------------


class Conv2D:
    """Cross-correlation convolution with a per-channel bias.

    Forward::

        Y[b, i, j, f] = bias[f] + sum_{u,v,c} X[b, i+u, j+v, c] * W[u, v, c, f]

    Backward (cross-correlation: the kernel is *not* conjugated)::

        dX[b, i, j, c] = sum_{u,v,f} dY[b, i+u, j+v, f] * W[u, v, c, f]
        dW[u, v, c, f] = sum_{b,i,j} dY[b, i, j, f] * X[b, i+u, j+v, c]
        db[f]          = sum_{b,i,j} dY[b, i, j, f]

    The weight gradient accumulates over all windows that cover a given
    ``(u, v, c)`` position; overlapping receptive fields therefore contribute
    additively, never overwritten.
    """

    def __init__(self, weight_shape: tuple[int, int, int, int], *, seed: int = 0) -> None:
        """Initialise weights and biases.

        Args:
            weight_shape: ``(KH, KW, Cin, Cout)``.
            seed: Seed for the normal generator used for the weights so that
                repeated tests get a reproducible array without sharing global
                state.
        """
        kh, kw, cin, cout = weight_shape
        if any(v <= 0 for v in (kh, kw, cin, cout)):
            raise ValueError(f"Conv2D weight shape must be positive, got {weight_shape}")
        rng = np.random.default_rng(seed)
        # He initialisation matches the project convention for the scratch
        # convolutional layer: variance 2 / fan_in, fan_in = kh * kw * cin.
        he_std = np.sqrt(2.0 / float(kh * kw * cin))
        self.W = (rng.standard_normal(weight_shape).astype(np.float64) * he_std)
        self.b = np.zeros((cout,), dtype=np.float64)
        self._cache: dict[str, Any] = {}

    # -- forward ------------------------------------------------------------

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Compute ``self.W * x + self.b`` (cross-correlation).

        Args:
            x: NHWC tensor with leading batch axis.

        Returns:
            ``(B, H - KH + 1, W - KW + 1, Cout)``.
        """
        self._cache = {"x": x}
        x = _as_batch(x)
        kh, kw, cin, cout = self.W.shape
        if cin != x.shape[-1]:
            raise ValueError(
                f"Conv2D: expected {cin} input channels, got {x.shape[-1]}"
            )
        batch = x.shape[0]
        h, w = x.shape[1], x.shape[2]
        out_h = h - kh + 1
        out_w = w - kw + 1
        if out_h <= 0 or out_w <= 0:
            raise ValueError(
                f"Conv2D output would be empty ({out_h}, {out_w}); "
                f"kernel {kh}x{kw} is too large for input spatial size ({h}, {w})"
            )

        y = np.empty((batch, out_h, out_w, cout), dtype=np.float64)
        for b in range(batch):
            for i in range(out_h):
                for j in range(out_w):
                    patch = x[b, i : i + kh, j : j + kw, :]  # (kh, kw, cin)
                    for f in range(cout):
                        y[b, i, j, f] = np.sum(patch * self.W[:, :, :, f]) + self.b[f]
        return y

    # -- backward -----------------------------------------------------------

    def backward(self, dy: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return ``(dX, dW, db)`` for this layer's cached forward input.

        Args:
            dy: Upstream gradient (lead batch axis).

        Returns:
            * ``dX``: same spatial shape as input ``x``
            * ``dW``: ``(KH, KW, Cin, Cout)``
            * ``db``: ``(Cout,)``
        """
        x = _as_batch(self._cache["x"])
        dy = _as_batch(dy)
        batch, h, w, cin = x.shape
        _, out_h, out_w, cout = dy.shape
        kh, kw, _, _ = self.W.shape
        if (out_h, out_w) != (h - kh + 1, w - kw + 1):
            raise ValueError(
                f"Conv2D backward: dy spatial shape {(out_h, out_w)} does not match "
                f"expected {(h - kh + 1, w - kw + 1)} for input {(h, w)} with kernel "
                f"({(kh, kw)})"
            )
        if cout != self.W.shape[3]:
            raise ValueError(f"Conv2D backward: dy channels {cout} != W channels {self.W.shape[3]}")

        dx = _zero(batch, h, w, cin)
        dw = _zero(kh, kw, cin, cout)
        db = _zero(cout)

        # dX[b, i, j, c] = sum_{u,v,f} dy[b, i-u, j-v, f] * W[u, v, c, f]
        # Cross-correlation backward: for each input position (i, j) and
        # channel c, add the upstream gradient of every output position
        # (i-u, j-v) that the kernel element W[u, v, c, f] feeds, then
        # multiply by that kernel element.  Only output positions that lie
        # inside the grid contribute.
        for b in range(batch):
            for i in range(h):
                for j in range(w):
                    for c in range(cin):
                        acc = 0.0
                        for u in range(kh):
                            ii = i - u
                            if ii < 0 or ii >= out_h:
                                continue
                            for v in range(kw):
                                jj = j - v
                                if jj < 0 or jj >= out_w:
                                    continue
                                for f in range(cout):
                                    acc += dy[b, ii, jj, f] * self.W[u, v, c, f]
                        dx[b, i, j, c] = acc

        # dW[u, v, c, f] = sum_{b,i,j} dy[b, i, j, f] * X[b, i+u, j+v, c]
        # Overlapping windows accumulate additively (hence +=).
        for b in range(batch):
            for i in range(out_h):
                for j in range(out_w):
                    for u in range(kh):
                        for v in range(kw):
                            for c in range(cin):
                                for f in range(cout):
                                    dw[u, v, c, f] += dy[b, i, j, f] * x[b, i + u, j + v, c]

        # db[f] = sum_{b,i,j} dy[b, i, j, f]
        for b in range(batch):
            for i in range(out_h):
                for j in range(out_w):
                    for f in range(cout):
                        db[f] += dy[b, i, j, f]

        return dx, dw, db


# ---------------------------------------------------------------------------
# ReLU
# ---------------------------------------------------------------------------


class ReLU:
    """Element-wise rectified linear unit.

    Forward::

        Y = max(0, X)

    Backward::

        dX = dY where X > 0, else 0

    The derivative at exactly zero is defined as **0** (the NumPy convention
    ``(X > 0)``), matching what Keras does.
    """

    def __init__(self) -> None:
        self._cache: dict[str, Any] = {}

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Apply ``max(0, x)`` element-wise. Preserves all but the leading axis."""
        x = _as_batch(x)
        self._cache["x"] = x
        return np.maximum(x, 0)

    def backward(self, dy: np.ndarray) -> np.ndarray:
        """Zero the gradient where the cached input was non-positive."""
        x = _as_batch(self._cache["x"])
        dy = _as_batch(dy)
        if dy.shape != x.shape:
            raise ValueError(
                f"ReLU backward: dy shape {dy.shape} != input shape {x.shape}"
            )
        mask = x > 0
        return np.where(mask, dy, np.zeros_like(dy))


# ---------------------------------------------------------------------------
# MaxPool2D
# ---------------------------------------------------------------------------


class MaxPool2D:
    """Max pooling with argmax routing of the gradient.

    Forward::

        Y[b, i, j, f] = max_{(u,v) in window} X[b, s*i+u, s*j+v, f]

    Backward::

        dX[b, s*i+u, s*j+v, f] += dY[b, i, j, f] for the winning (u, v)

    The output spatial size is ``floor((H - pool_size) / stride) + 1`` per
    axis (valid pooling: trailing rows/columns that cannot fill a whole
    window are dropped).  The gradients are summed with ``+=`` so that
    overlapping windows (``stride < pool_size``) sharing an input cell each
    contribute.  Ties resolve to the **first maximum in row-major order**
    (smallest ``(u, v)`` lexicographically); this is exactly what
    ``numpy.argmax`` does on a flattened window, so the forward pass and the
    gradient routing agree.
    """

    def __init__(self, pool_size: int = 2, stride: int | None = None) -> None:
        if pool_size <= 0:
            raise ValueError("pool_size must be positive")
        self.pool_size = pool_size
        self.stride = stride if stride is not None else pool_size
        if self.stride <= 0:
            raise ValueError("stride must be positive")
        self._cache: dict[str, Any] = {}

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Pool along the two trailing spatial axes. Caches argmax positions."""
        x = _as_batch(x)
        batch, h, w, cin = x.shape
        if h < self.pool_size or w < self.pool_size:
            raise ValueError(
                f"MaxPool2D input ({h}, {w}) is smaller than the "
                f"{self.pool_size}x{self.pool_size} window"
            )
        out_h = (h - self.pool_size) // self.stride + 1
        out_w = (w - self.pool_size) // self.stride + 1

        y = np.empty((batch, out_h, out_w, cin), dtype=np.float64)
        # argmax[b, i, j, c] = flat spatial index (row-major over h x w) of
        # the winning input cell for that output window and channel.
        argmax = np.zeros((batch, out_h, out_w, cin), dtype=np.int64)
        for b in range(batch):
            for i in range(out_h):
                for j in range(out_w):
                    window = x[b, i * self.stride : i * self.stride + self.pool_size,
                               j * self.stride : j * self.stride + self.pool_size, :]
                    for c in range(cin):
                        flat = window[:, :, c].reshape(-1)
                        best = int(np.argmax(flat))
                        local_u, local_v = divmod(best, self.pool_size)
                        y[b, i, j, c] = window[local_u, local_v, c]
                        argmax[b, i, j, c] = (
                            (i * self.stride + local_u) * w
                            + (j * self.stride + local_v)
                        )
        self._cache = {"x": x, "argmax": argmax}
        return y

    def backward(self, dy: np.ndarray) -> np.ndarray:
        """Route each gradient to the cached maximum position of its window."""
        x = _as_batch(self._cache["x"])
        dy = _as_batch(dy)
        batch, h, w, cin = x.shape
        _, out_h, out_w, _ = dy.shape
        expected = ((h - self.pool_size) // self.stride + 1,
                    (w - self.pool_size) // self.stride + 1)
        if (out_h, out_w) != expected:
            raise ValueError(
                f"MaxPool2D backward: dy spatial shape {(out_h, out_w)} does not match "
                f"expected {expected}"
            )
        if dy.shape[3] != cin:
            raise ValueError(f"MaxPool2D backward: dy channels {dy.shape[3]} != {cin}")

        dx = np.zeros_like(x, dtype=np.float64)
        dx_flat = dx.reshape(batch, h * w, cin)
        argmax = self._cache["argmax"]
        for b in range(batch):
            for i in range(out_h):
                for j in range(out_w):
                    for c in range(cin):
                        # Overlapping windows may share an input cell, so the
                        # routed gradients are summed, not overwritten.
                        dx_flat[b, argmax[b, i, j, c], c] += dy[b, i, j, c]
        return dx


# ---------------------------------------------------------------------------
# Flatten
# ---------------------------------------------------------------------------


class Flatten:
    """Row-major reshape of the trailing axes to a single feature vector.

    The leading batch axis is preserved exactly; the output shape is
    ``(B, H*W*C)`` and the backward pass restores the cached input shape.
    """

    def __init__(self) -> None:
        self._cache: dict[str, Any] = {}

    def forward(self, x: np.ndarray) -> np.ndarray:
        x = _as_batch(x)
        self._cache["shape"] = x.shape
        return x.reshape(x.shape[0], -1)

    def backward(self, dy: np.ndarray) -> np.ndarray:
        original_shape = self._cache["shape"]
        # The flattened axis holds exactly the per-sample features:
        # height * width * channels (the batch axis is preserved as rows).
        expected_features = int(np.prod(original_shape[1:]))
        if dy.ndim != 2 or dy.shape[1] != expected_features:
            raise ValueError(
                f"Flatten backward: dy has shape {dy.shape}, "
                f"expected (batch, {expected_features})"
            )
        return dy.reshape(*original_shape)


# ---------------------------------------------------------------------------
# Dense
# ---------------------------------------------------------------------------


class Dense:
    """Fully connected layer with a linear activation.

    Forward::

        Z = X @ W + b

    Backward (batch loss already divided by the batch size exactly once, so the
    parameter gradients are **not** divided again)::

        dX = dZ @ W.T
        dW = X.T @ dZ
        db = sum(dZ, axis=0)
    """

    def __init__(self, in_features: int, out_features: int, *, seed: int = 0) -> None:
        if in_features <= 0 or out_features <= 0:
            raise ValueError(
                f"Dense layers need positive sizes, got in={in_features}, out={out_features}"
            )
        rng = np.random.default_rng(seed)
        he_std = np.sqrt(2.0 / float(in_features))
        self.W = (rng.standard_normal((in_features, out_features)).astype(np.float64) * he_std)
        self.b = np.zeros((out_features,), dtype=np.float64)
        self._cache: dict[str, Any] = {}

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Linear map ``Z = X @ W + b``. Preserves the leading batch axis."""
        x = _as_batch(x)
        self._cache = {"x": x}
        if x.shape[1] != self.W.shape[0]:
            raise ValueError(
                f"Dense: expected {self.W.shape[0]} inputs, got {x.shape[1]}"
            )
        return x @ self.W + self.b

    def backward(self, dy: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return ``(dX, dW, db)``."""
        x = _as_batch(self._cache["x"])
        dy = _as_batch(dy)
        if dy.shape != (x.shape[0], self.W.shape[1]):
            raise ValueError(
                f"Dense backward: dy shape {dy.shape}, expected ({x.shape[0]}, {self.W.shape[1]})"
            )
        dx = dy @ self.W.T
        dw = x.T @ dy
        db = dy.sum(axis=0)
        return dx, dw, db


# ---------------------------------------------------------------------------
# Stable softmax
# ---------------------------------------------------------------------------


class StableSoftmax:
    """Numerically stable softmax over the trailing (class) axis.

    Forward::

        m = max_k Z[..., k]
        P = exp(Z - m) / sum_k exp(Z - m)

    The shift keeps the largest numerator at exactly 1 (no overflow) and the
    smallest probability limited only by float32 subnormals, as derived in
    Section 12 of ``docs/cnn_mathematical_foundations.md``.

    Backward contract
    -----------------
    ``StableSoftmax.backward(dP)`` implements the **general softmax
    vector-Jacobian product**.  For an arbitrary upstream gradient ``dP``::

        dZ = P * (dP - sum(dP * P, axis=-1, keepdims=True))

    It is **not** the fused softmax+cross-entropy shortcut.  The combined
    softmax + categorical cross-entropy path, whose logits gradient is
    ``(P - Y) / B`` (derived in Section 14), is a *separate* explicit
    computation performed by :class:`CategoricalCrossEntropy.backward`
    (``(P - Y) / B``) and by :class:`ScratchCNN.backward` (which hands this
    value to the dense layer directly, without passing it through the
    softmax).  Applying the softmax derivative a second time to an already-
    fused logits gradient would double-count the softmax and corrupt the
    gradient.
    """

    def __init__(self) -> None:
        self._cache: dict[str, Any] = {}

    def forward(self, z: np.ndarray) -> np.ndarray:
        """Compute the stable softmax over the last axis."""
        z = _as_batch(z)
        self._cache["z"] = z
        z = z - np.max(z, axis=-1, keepdims=True)
        e = np.exp(z)
        p = e / np.sum(e, axis=-1, keepdims=True)
        self._cache["p"] = p
        return p

    def backward(self, dP: np.ndarray) -> np.ndarray:
        """Gradient of the softmax for an arbitrary upstream gradient.

        Vector-Jacobian product ``dZ = P * (dP - sum(dP * P, axis=-1))``.

        Args:
            dP: Upstream gradient, same shape as the logits ``(B, N)``.

        Returns:
            ``dZ``: gradient w.r.t. the logits.
        """
        z = _as_batch(self._cache["z"])
        dP = _as_batch(dP)
        if dP.shape != z.shape:
            raise ValueError(
                f"StableSoftmax backward: dP shape {dP.shape} != input shape {z.shape}"
            )
        p = self._cache.get("p")
        if p is None:
            z = z - np.max(z, axis=-1, keepdims=True)
            e = np.exp(z)
            p = e / np.sum(e, axis=-1, keepdims=True)
        return p * (dP - np.sum(dP * p, axis=-1, keepdims=True))


# ---------------------------------------------------------------------------
# Categorical cross-entropy
# ---------------------------------------------------------------------------


class CategoricalCrossEntropy:
    """Mean categorical cross-entropy over one-hot labels.

    L = -(1 / B) * sum_b sum_c Y[b, c] * log(P[b, c])

    Probabilities are clipped to ``[eps, 1]`` before the log, so a degenerate
    probability of exactly 0 never turns the loss into ``inf``.

    Backward contract
    -----------------
    ``CategoricalCrossEntropy.backward()`` returns the gradient of the mean
    loss with respect to the **pre-softmax logits**, i.e. the fused
    softmax+cross-entropy logits gradient::

        G = (P - Y) / B

    This is derived in Section 14 of
    ``docs/cnn_mathematical_foundations.md``: chaining the softmax Jacobian
    with the cross-entropy derivative collapses to ``(P - Y) / B``.  This is a
    single application of the batch averaging (at the loss level); the
    softmax layer's backward must not be applied a second time.
    """

    def __init__(self, eps: float = 1e-12) -> None:
        self.eps = eps
        self._cache: dict[str, Any] = {}

    def forward(self, p: np.ndarray, y: np.ndarray) -> float:
        """Compute the mean cross-entropy over the batch.

        Args:
            p: Probabilities ``(B, N)``.
            y: One-hot targets ``(B, N)``.

        Returns:
            The scalar mean loss (Python float).
        """
        p = np.asarray(p, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        if p.ndim != 2 or y.ndim != 2:
            raise ValueError(f"expected 2-D (B, N), got p={p.shape}, y={y.shape}")
        if p.shape != y.shape:
            raise ValueError(f"p and y must have the same shape, got {p.shape} vs {y.shape}")
        self._cache = {"p": p, "y": y}
        return float(-np.sum(y * np.log(np.clip(p, self.eps, 1.0))) / p.shape[0])

    def backward(self) -> np.ndarray:
        """Gradient of the mean loss w.r.t. the pre-softmax logits.

        Returns ``(P - Y) / B`` element-wise, with the same shape as ``p``.
        The batch mean gives one division by ``B``; no softmax derivative is
        applied here.
        """
        p = self._cache["p"]
        y = self._cache["y"]
        return (p - y) / p.shape[0]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

__all__ = [
    "Conv2D",
    "ReLU",
    "MaxPool2D",
    "Flatten",
    "Dense",
    "StableSoftmax",
    "CategoricalCrossEntropy",
]
