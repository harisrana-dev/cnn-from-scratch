# Project Architecture

## Purpose

Compare two implementations of the same convolutional neural network on the same
data:

1. **Scratch model** – Python + NumPy. Convolution, activation, pooling, dense
   layers, softmax, loss, forward propagation, backpropagation and gradient
   descent are all written by hand.
2. **Framework model** – Keras/TensorFlow, built from the equivalent layer
   specification, used as the reference implementation.

The comparison is only meaningful if both models receive identical inputs and
implement the same architecture; the shared dataset artefacts in
`data/processed/` guarantee the first condition.

## What TensorFlow/Keras may and may not be used for

| Concern | Scratch CNN | Keras CNN |
|---|---|---|
| Dataset download | allowed (`keras.datasets`) | allowed |
| Convolution / pooling / dense layers | **must be NumPy** | Keras layers |
| Activation and softmax | **must be NumPy** | Keras activations |
| Loss function | **must be NumPy** | Keras loss |
| Forward propagation | **must be NumPy** | Keras `model.fit` |
| Backpropagation and gradients | **must be NumPy** | autodiff |
| Weight updates (SGD) | **must be NumPy** | Keras optimizer |
| Data loading of the processed `.npz` | NumPy `np.load` | NumPy `np.load` |

TensorFlow is never imported by `cnn_scratch.models` once implemented, and never
computes the scratch model's forward pass, backward pass or parameter updates.
`src/cnn_scratch/data/prepare_data.py` imports Keras inside
`load_fashion_mnist()` only, and that import is documented as an acquisition
exception.

## Layer overview

```
src/cnn_scratch/
├── __init__.py            package metadata, no heavyweight imports
├── data/
│   ├── __init__.py        re-exports constants and I/O helpers
│   ├── dataset_utils.py   constants, validation, NPZ/JSON persistence
│   └── prepare_data.py    acquisition → subset → normalise → validate → save
├── models/
│   └── __init__.py        placeholder: layers/models arrive in Phase 3+
└── utils/
    ├── __init__.py
    └── visualization.py   sample grid and class-distribution figures
```

### Responsibilities

`dataset_utils.py` knows nothing about models: it owns path resolution, dataset
constants (28x28x1, 10 classes, class names), `class_distribution()`,
`validate_dataset()`, and the `save_dataset`/`load_dataset`/`save_metadata`/
`load_metadata` helpers. Keeping validation here means both implementations
consume data that satisfies the same invariants.

`prepare_data.py` owns the pipeline itself. Its functions are deliberately
separate so each step can be demonstrated and tested individually:

| Function | Responsibility |
|---|---|
| `load_fashion_mnist()` | download/load official splits, validate their shapes |
| `select_balanced_subset()` | seeded, per-class sampling without replacement |
| `normalize_images()` | float32, divide by 255, append channel axis |
| `build_metadata()` | assemble the metadata record |
| `prepare_dataset()` | orchestrate the pipeline and persist artefacts |
| `main()` | CLI entry point (`python -m cnn_scratch.data.prepare_data`) |

`visualization.py` renders figures to `results/figures/` with the Agg backend so
plots work in a terminal and in VS Code alike.

## Data flow

```
keras.datasets.fashion_mnist
        │  (uint8, (60000, 28, 28) / (10000, 28, 28))
        ▼
select_balanced_subset(seed=42, 1000/class | 200/class)
        ▼
normalize_images()                float32, [0,1], (n, 28, 28, 1)
        ▼
validate_dataset()                shapes, dtype, range, balance
        ▼
data/processed/fashion_mnist_subset.npz
data/processed/fashion_mnist_subset_metadata.json
        ▼
scratch CNN (NumPy)          Keras CNN (TensorFlow)
```

## Design decisions

- **Flat, explicit modules** instead of a framework: the code must be
  explainable in the assignment viva.
- **Single source of truth for constants** in `dataset_utils.py`; the seed and
  subset sizes are read from `configs/experiment_config.json`.
- **Relative paths only**, resolved from `__file__`, so the repository works on
  any machine and from any working directory.
- **Compressed `.npz`** for the processed subsets: small, dependency-free and
  readable with `np.load`.
- **No packaging metadata**: `conftest.py` puts `src/` on `sys.path`, which is
  enough for an assignment and avoids an unnecessary install step.

## Planned additions (not implemented)

| File | Phase | Content |
|---|---|---|
| `models/layers.py` | 3 | `Conv2D`, `ReLU`, `MaxPool2D`, `Dense`, `Softmax` with `forward`/`backward` |
| `models/scratch_cnn.py` | 4 | parameter init, forward, backward, SGD, `predict`, `evaluate` |
| `models/keras_cnn.py` | 5 | equivalent Keras model |
| `utils/metrics.py` | 5 | accuracy, confusion matrix, history export |
| `utils/reporting.py` | 6 | comparison tables and figures |
