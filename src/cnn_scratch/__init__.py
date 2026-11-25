"""Convolutional neural networks from scratch, compared with Keras/TensorFlow.

The package is organised so that the NumPy implementation and the framework
implementation can be inspected side by side:

* ``cnn_scratch.data`` – dataset acquisition, subset selection, preprocessing.
* ``cnn_scratch.models`` – CNN layer and model implementations (later phases).
* ``cnn_scratch.utils`` – figure and reporting helpers.

Importing this package never imports TensorFlow or Keras.
"""

__version__ = "0.1.0"
