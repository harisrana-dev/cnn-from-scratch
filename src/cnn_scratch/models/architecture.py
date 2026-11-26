"""The frozen CNN architecture: shape and parameter arithmetic only.

Both implementations build the network declared in
``configs/experiment_config.json``:

.. code-block:: text

    Input      (28, 28, 1)
    Conv2D     8 filters, 3x3, stride 1, valid padding, bias   -> (26, 26, 8)
    ReLU       element-wise                                     -> (26, 26, 8)
    MaxPool2D  2x2, stride 2, valid padding                      -> (13, 13, 8)
    Flatten                                                      -> (1352,)
    Dense      10 units, bias, raw logits                        -> (10,)
    Softmax    10 class probabilities                            -> (10,)

This module contains **no layer implementations**: no convolution, activation,
pooling, loss or gradient code. It validates a layer specification and computes
each layer's output shape and trainable-parameter count, so the numbers quoted
in ``docs/cnn_mathematical_foundations.md`` and
``docs/backpropagation_walkthrough.md`` are computed rather than copied by hand.
Forward and backward passes belong to Phase 3.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from cnn_scratch.data.dataset_utils import (
    CHANNELS,
    IMAGE_HEIGHT,
    IMAGE_WIDTH,
    load_experiment_config,
)

#: Layer types the frozen architecture may contain.
SUPPORTED_LAYERS: tuple[str, ...] = (
    "conv2d",
    "relu",
    "maxpool2d",
    "flatten",
    "dense",
    "softmax",
)

#: Padding modes implemented by the project. ``same`` is deliberately excluded:
#: the frozen architecture can be derived with ``valid`` convolution only.
SUPPORTED_PADDING: tuple[str, ...] = ("valid",)

DEFAULT_INPUT_SHAPE: tuple[int, int, int] = (IMAGE_HEIGHT, IMAGE_WIDTH, CHANNELS)


class ArchitectureError(ValueError):
    """Raised when an architecture specification is invalid or inconsistent."""


@dataclass(frozen=True)
class LayerInfo:
    """Everything Phase 2 can state about one layer without implementing it.

    Attributes:
        name: Layer type, e.g. ``"conv2d"``.
        output_shape: Shape of one sample (no batch dimension).
        parameters: Trainable parameters the layer contributes.
    """

    name: str
    output_shape: tuple[int, ...]
    parameters: int


# --------------------------------------------------------------------------- #
# Elementary size arithmetic
# --------------------------------------------------------------------------- #
def conv_output_size(size: int, kernel: int, stride: int = 1, padding: int = 0) -> int:
    """Output length along one spatial axis of a convolution or pooling window.

    ``out = floor((size + 2 * padding - kernel) / stride) + 1``

    Args:
        size: Input length along the axis.
        kernel: Kernel (or pooling window) length.
        stride: Step between window positions.
        padding: Implicit zeros added to each side.

    Returns:
        The output length.

    Raises:
        ArchitectureError: If the arguments cannot produce a non-empty output.
    """
    if kernel <= 0:
        raise ArchitectureError(f"kernel size must be positive, got {kernel}")
    if stride <= 0:
        raise ArchitectureError(f"stride must be positive, got {stride}")
    if padding < 0:
        raise ArchitectureError(f"padding must be non-negative, got {padding}")

    out = (size - kernel + 2 * padding) // stride + 1
    if out <= 0:
        raise ArchitectureError(
            f"a kernel of {kernel} with stride {stride} and padding {padding} "
            f"does not fit an input of size {size}"
        )
    return out


def batch_shape(shape: Sequence[int], batch_size: int = 32) -> tuple[int, ...]:
    """Prepend the batch dimension to a per-sample shape."""
    if batch_size <= 0:
        raise ArchitectureError(f"batch size must be positive, got {batch_size}")
    return (int(batch_size), *(int(dimension) for dimension in shape))


# --------------------------------------------------------------------------- #
# Specification helpers
# --------------------------------------------------------------------------- #
def _positive_int(layer: Mapping[str, Any], key: str, index: int, default: int | None = None) -> int:
    value = layer.get(key, default)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ArchitectureError(
            f"layer {index} ({layer.get('type')!r}): {key!r} must be an integer, got {value!r}"
        )
    if value <= 0:
        raise ArchitectureError(
            f"layer {index} ({layer.get('type')!r}): {key!r} must be positive, got {value}"
        )
    return value


def _flag(layer: Mapping[str, Any], key: str, index: int, default: bool = True) -> bool:
    value = layer.get(key, default)
    if not isinstance(value, bool):
        raise ArchitectureError(
            f"layer {index} ({layer.get('type')!r}): {key!r} must be a boolean, got {value!r}"
        )
    return value


def _image_shape(shape: tuple[int, ...], kind: str, index: int) -> tuple[int, int, int]:
    if len(shape) != 3:
        raise ArchitectureError(
            f"layer {index} ({kind!r}) expects a 3-D (H, W, C) input, got shape {shape}"
        )
    return shape  # type: ignore[return-value]


def _vector_length(shape: tuple[int, ...], kind: str, index: int) -> int:
    if len(shape) != 1:
        raise ArchitectureError(
            f"layer {index} ({kind!r}) expects a flattened 1-D input, got shape {shape}"
        )
    return shape[0]


# --------------------------------------------------------------------------- #
# Architecture description
# --------------------------------------------------------------------------- #
def describe_architecture(
    layers: Sequence[Mapping[str, Any]],
    input_shape: Sequence[int] = DEFAULT_INPUT_SHAPE,
) -> tuple[LayerInfo, ...]:
    """Validate a layer specification and describe every layer.

    Args:
        layers: Layer mappings, each with at least a ``"type"`` key.
        input_shape: Per-sample input shape, ``(H, W, C)``.

    Returns:
        One :class:`LayerInfo` per layer, in forward order.

    Raises:
        ArchitectureError: If any layer is unsupported, malformed or
            dimensionally inconsistent with the layers before it.
    """
    if not layers:
        raise ArchitectureError("the architecture must declare at least one layer")

    shape = tuple(int(dimension) for dimension in input_shape)
    if len(shape) != 3 or any(dimension <= 0 for dimension in shape):
        raise ArchitectureError(
            f"input shape must be three positive dimensions (H, W, C), got {shape}"
        )

    described: list[LayerInfo] = []

    for index, layer in enumerate(layers):
        if not isinstance(layer, Mapping):
            raise ArchitectureError(
                f"layer {index} must be a mapping, got {type(layer).__name__}"
            )

        kind = layer.get("type")
        if kind not in SUPPORTED_LAYERS:
            raise ArchitectureError(
                f"layer {index}: unsupported layer type {kind!r}; "
                f"expected one of {SUPPORTED_LAYERS}"
            )

        if described and described[-1].name == "softmax":
            raise ArchitectureError(f"layer {index}: softmax must be the final layer")

        if kind == "conv2d":
            filters = _positive_int(layer, "filters", index)
            kernel = _positive_int(layer, "kernel_size", index)
            stride = _positive_int(layer, "stride", index, default=1)
            padding = layer.get("padding", "valid")
            if padding not in SUPPORTED_PADDING:
                raise ArchitectureError(
                    f"layer {index} (conv2d): padding {padding!r} is not supported; "
                    f"expected one of {SUPPORTED_PADDING}"
                )
            use_bias = _flag(layer, "use_bias", index)

            height, width, channels = _image_shape(shape, kind, index)
            out_height = conv_output_size(height, kernel, stride)
            out_width = conv_output_size(width, kernel, stride)

            weights = kernel * kernel * channels * filters
            parameters = weights + filters if use_bias else weights
            shape = (out_height, out_width, filters)

        elif kind == "relu":
            if len(shape) != 3:
                raise ArchitectureError(
                    f"layer {index} (relu) expects a 3-D (H, W, C) input, got shape {shape}"
                )
            parameters = 0

        elif kind == "maxpool2d":
            pool_size = _positive_int(layer, "pool_size", index)
            stride = _positive_int(layer, "stride", index, default=pool_size)
            padding = layer.get("padding", "valid")
            if padding not in SUPPORTED_PADDING:
                raise ArchitectureError(
                    f"layer {index} (maxpool2d): padding {padding!r} is not supported; "
                    f"expected one of {SUPPORTED_PADDING}"
                )

            height, width, channels = _image_shape(shape, kind, index)
            out_height = conv_output_size(height, pool_size, stride)
            out_width = conv_output_size(width, pool_size, stride)
            shape = (out_height, out_width, channels)
            parameters = 0

        elif kind == "flatten":
            height, width, channels = _image_shape(shape, kind, index)
            shape = (height * width * channels,)
            parameters = 0

        elif kind == "dense":
            units = _positive_int(layer, "units", index)
            use_bias = _flag(layer, "use_bias", index)
            inputs = _vector_length(shape, kind, index)
            parameters = inputs * units + (units if use_bias else 0)
            shape = (units,)

        else:  # softmax
            _vector_length(shape, kind, index)
            parameters = 0

        described.append(LayerInfo(name=str(kind), output_shape=shape, parameters=parameters))

    if described[-1].name != "softmax":
        raise ArchitectureError(
            "the frozen architecture must end with a softmax layer, "
            f"got {described[-1].name!r}"
        )

    return tuple(described)


def parameter_breakdown(described: Sequence[LayerInfo]) -> dict[str, int]:
    """Trainable parameters per layer, omitting layers that own none."""
    return {layer.name: layer.parameters for layer in described if layer.parameters}


def total_parameters(described: Sequence[LayerInfo]) -> int:
    """Total trainable parameters of a described architecture."""
    return sum(layer.parameters for layer in described)


# --------------------------------------------------------------------------- #
# Loading the frozen specification from the experiment configuration
# --------------------------------------------------------------------------- #
def architecture_spec(
    config: Mapping[str, Any] | None = None,
) -> tuple[tuple[int, int, int], list[Mapping[str, Any]]]:
    """Read the architecture block from the experiment configuration.

    Args:
        config: Parsed configuration; loaded from
            ``configs/experiment_config.json`` when omitted.

    Returns:
        ``(input_shape, layers)``.

    Raises:
        ArchitectureError: If the configuration has no usable architecture block.
    """
    settings = dict(config) if config is not None else load_experiment_config()
    block = settings.get("architecture")
    if not isinstance(block, Mapping):
        raise ArchitectureError("experiment configuration has no 'architecture' object")

    input_shape = tuple(int(dimension) for dimension in block.get("input_shape", DEFAULT_INPUT_SHAPE))
    if len(input_shape) != 3:
        raise ArchitectureError(
            f"'architecture.input_shape' must have three dimensions, got {input_shape}"
        )

    layers = block.get("layers")
    if not isinstance(layers, list) or not layers:
        raise ArchitectureError("'architecture.layers' must be a non-empty list")

    return input_shape, layers


def frozen_architecture(
    config: Mapping[str, Any] | None = None,
) -> tuple[LayerInfo, ...]:
    """Describe the architecture declared by the experiment configuration."""
    input_shape, layers = architecture_spec(config)
    return describe_architecture(layers, input_shape)
