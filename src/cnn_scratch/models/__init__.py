"""CNN implementations.

Implemented:

* ``architecture.py`` – the frozen architecture from
  ``configs/experiment_config.json``: per-layer output shapes and trainable
  parameter counts, with validation. No layer computation happens here.
* ``layers.py`` (Phase 3) – NumPy convolution, ReLU, max-pooling, flatten,
  dense, stable softmax and categorical cross-entropy layers with manual
  forward and backward passes (pure NumPy, no autograd).
* ``scratch_cnn.py`` (Phase 3) – the scratch model: layer assembly, forward
  propagation and backpropagation returning every gradient. Parameter
  updates, metrics and the training loop belong to Phase 4.

Planned:

* ``keras_cnn.py`` – the equivalent Keras/TensorFlow model built from the
  same layer specification for comparison (Phase 5).
"""
