"""Tests for the frozen CNN architecture and its shape/parameter arithmetic.

These tests validate the architecture declared in
``configs/experiment_config.json``. They exercise no layer computation: nothing
in Phase 2 performs a forward or backward pass.
"""

from __future__ import annotations

import pytest

from cnn_scratch.data.dataset_utils import (
    CHANNELS,
    IMAGE_HEIGHT,
    IMAGE_WIDTH,
    NUM_CLASSES,
    load_experiment_config,
)
from cnn_scratch.models.architecture import (
    ArchitectureError,
    architecture_spec,
    batch_shape,
    conv_output_size,
    describe_architecture,
    frozen_architecture,
    parameter_breakdown,
    total_parameters,
)

FROZEN_LAYER_NAMES = ("conv2d", "relu", "maxpool2d", "flatten", "dense", "softmax")
FROZEN_OUTPUT_SHAPES = (
    (26, 26, 8),
    (26, 26, 8),
    (13, 13, 8),
    (1352,),
    (10,),
    (10,),
)
CONV_PARAMETERS = 80
DENSE_PARAMETERS = 13_530
TOTAL_PARAMETERS = 13_610

BATCH_SIZE = 32


@pytest.fixture(scope="module")
def described():
    """The frozen architecture as described from the configuration file."""
    return frozen_architecture()


def spec_with(**overrides) -> list[dict]:
    """Frozen layer list with one layer's fields overridden."""
    _, layers = architecture_spec()
    layers = [dict(layer) for layer in layers]
    for position, changes in overrides.items():
        layers[int(position)].update(changes)
    return layers


# --------------------------------------------------------------------------- #
# Shapes
# --------------------------------------------------------------------------- #
def test_frozen_layer_sequence(described):
    assert tuple(layer.name for layer in described) == FROZEN_LAYER_NAMES


def test_frozen_layer_output_shapes(described):
    assert tuple(layer.output_shape for layer in described) == FROZEN_OUTPUT_SHAPES


def test_flatten_dimension_is_1352(described):
    flatten = described[3]
    assert flatten.name == "flatten"
    assert flatten.output_shape == (1352,)
    assert 13 * 13 * 8 == 1352


def test_batch_dimension_is_prepended(described):
    assert batch_shape(described[0].output_shape, BATCH_SIZE) == (32, 26, 26, 8)
    assert batch_shape(described[3].output_shape, BATCH_SIZE) == (32, 1352)
    assert batch_shape(described[-1].output_shape) == (32, 10)


def test_input_shape_matches_the_dataset(described):
    input_shape, _ = architecture_spec()
    assert input_shape == (IMAGE_HEIGHT, IMAGE_WIDTH, CHANNELS)


def test_output_dimension_formula():
    assert conv_output_size(28, 3, 1) == 26
    assert conv_output_size(26, 2, 2) == 13
    assert conv_output_size(28, 3, 2) == 13
    assert conv_output_size(5, 3, 1, padding=1) == 5
    assert conv_output_size(7, 7, 7) == 1


# --------------------------------------------------------------------------- #
# Parameters
# --------------------------------------------------------------------------- #
def test_convolution_parameter_count(described):
    conv = described[0]
    assert conv.parameters == CONV_PARAMETERS
    assert (3 * 3 * 1 + 1) * 8 == CONV_PARAMETERS


def test_dense_parameter_count(described):
    dense = described[4]
    assert dense.parameters == DENSE_PARAMETERS
    assert 1352 * 10 + 10 == DENSE_PARAMETERS


def test_total_trainable_parameters(described):
    assert total_parameters(described) == TOTAL_PARAMETERS
    assert CONV_PARAMETERS + DENSE_PARAMETERS == TOTAL_PARAMETERS


def test_parameter_breakdown_is_only_conv_and_dense(described):
    assert parameter_breakdown(described) == {"conv2d": 80, "dense": 13_530}


def test_parameterless_layers_contribute_nothing(described):
    for layer in described:
        if layer.name in {"relu", "maxpool2d", "flatten", "softmax"}:
            assert layer.parameters == 0


def test_bias_flags_change_parameter_counts():
    described = describe_architecture(spec_with(**{"0": {"use_bias": False}}))
    assert described[0].parameters == 72  # (3*3*1)*8
    assert total_parameters(described) == 72 + DENSE_PARAMETERS

    described = describe_architecture(spec_with(**{"4": {"use_bias": False}}))
    assert described[4].parameters == 13_520  # 1352*10
    assert total_parameters(described) == CONV_PARAMETERS + 13_520


# --------------------------------------------------------------------------- #
# Configuration consistency
# --------------------------------------------------------------------------- #
def test_configuration_records_frozen_architecture():
    _, layers = architecture_spec()
    conv, relu, pool, flatten, dense, softmax = (dict(layer) for layer in layers)

    assert conv["filters"] == 8
    assert conv["kernel_size"] == 3
    assert conv["stride"] == 1
    assert conv["padding"] == "valid"
    assert conv["use_bias"] is True

    assert relu["type"] == "relu"

    assert pool["pool_size"] == 2
    assert pool["stride"] == 2
    assert pool["padding"] == "valid"

    assert flatten["type"] == "flatten"

    assert dense["units"] == NUM_CLASSES
    assert dense["use_bias"] is True
    assert dense["activation"] == "linear"

    assert softmax["type"] == "softmax"


def test_configuration_records_shared_training_settings():
    config = load_experiment_config()

    assert config["random_seed"] == 42
    assert config["image_layout"] == "NHWC"
    assert config["batch_size"] == 32
    assert config["initial_epochs"] == 5
    assert config["optimizer"] == "SGD"
    assert config["learning_rate"] == 0.01
    assert config["loss"] == "categorical_crossentropy"
    assert config["loss_reduction"] == "mean"
    assert config["initialization"] == {"weights": "he_normal", "biases": "zeros"}


def test_configuration_architecture_matches_dataset_settings():
    config = load_experiment_config()
    input_shape, _ = architecture_spec(config)

    assert input_shape == (config["image_height"], config["image_width"], config["channels"])
    assert config["num_classes"] == NUM_CLASSES


# --------------------------------------------------------------------------- #
# Invalid specifications are rejected
# --------------------------------------------------------------------------- #
def test_rejects_empty_layer_list():
    with pytest.raises(ArchitectureError, match="at least one layer"):
        describe_architecture([])


def test_rejects_unknown_layer_type():
    layers = spec_with(**{"1": {"type": "batchnorm"}})
    with pytest.raises(ArchitectureError, match="unsupported layer type"):
        describe_architecture(layers)


def test_rejects_non_valid_padding():
    layers = spec_with(**{"0": {"padding": "same"}})
    with pytest.raises(ArchitectureError, match="padding 'same' is not supported"):
        describe_architecture(layers)


def test_rejects_kernel_larger_than_input():
    layers = spec_with(**{"0": {"kernel_size": 29}})
    with pytest.raises(ArchitectureError, match="does not fit an input of size 28"):
        describe_architecture(layers)


def test_rejects_non_positive_stride():
    layers = spec_with(**{"0": {"stride": 0}})
    with pytest.raises(ArchitectureError, match="'stride' must be positive"):
        describe_architecture(layers)


def test_rejects_boolean_integer_configuration():
    layers = spec_with(**{"0": {"filters": True}})
    with pytest.raises(ArchitectureError, match="'filters' must be an integer"):
        describe_architecture(layers)


def test_rejects_dense_before_flatten():
    layers = spec_with(**{"3": {"type": "dense", "units": 10}})
    with pytest.raises(ArchitectureError, match="flattened 1-D input"):
        describe_architecture(layers)


def test_rejects_double_flatten():
    layers = spec_with(**{"4": {"type": "flatten"}})
    with pytest.raises(ArchitectureError, match="3-D"):
        describe_architecture(layers)


def test_rejects_missing_softmax():
    layers = [dict(layer) for layer in architecture_spec()[1][:-1]]
    with pytest.raises(ArchitectureError, match="must end with a softmax layer"):
        describe_architecture(layers)


def test_rejects_layers_after_softmax():
    layers = spec_with()
    layers.append({"type": "dense", "units": 10})
    with pytest.raises(ArchitectureError, match="softmax must be the final layer"):
        describe_architecture(layers)


def test_alternative_kernel_size_propagates_through_shapes():
    layers = spec_with(**{"0": {"kernel_size": 5, "stride": 1}})
    described = describe_architecture(layers)
    assert described[0].output_shape == (24, 24, 8)
    assert described[2].output_shape == (12, 12, 8)
    assert described[3].output_shape == (1152,)
