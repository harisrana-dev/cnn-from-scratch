"""CNN implementations (populated in later phases).

Implemented:

* ``architecture.py`` – the frozen architecture from
  ``configs/experiment_config.json``: per-layer output shapes and trainable
  parameter counts, with validation. No layer computation happens here.

Planned, not yet implemented:

* ``layers.py`` – NumPy convolution, ReLU, max-pooling, dense and softmax layers
  with manual forward and backward passes.
* ``scratch_cnn.py`` – the scratch model: parameter initialisation, forward
  propagation, backpropagation, SGD updates and prediction.
* ``keras_cnn.py`` – the equivalent Keras/TensorFlow model built from the same
  layer specification for comparison.

No forward pass, backward pass, loss, gradient or training code exists yet;
Phases 1 and 2 cover the dataset, the mathematics and the architecture only.
"""
