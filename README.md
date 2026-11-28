# Convolutional Neural Networks: Manual NumPy Implementation vs. Keras/TensorFlow

A comparative implementation of convolutional neural networks using NumPy and
Keras/TensorFlow, demonstrating convolution operations, activation functions,
pooling, forward propagation, backpropagation, gradient descent, and image
classification using Fashion-MNIST.

This repository is an individual university assignment in Artificial
Intelligence / Deep Learning. It is written to be read: every operation the
scratch model performs is plain, inspectable Python.

**Current status: Phase 3 complete** — repository foundation, environment,
dataset pipeline, mathematical foundations, the validated shared architecture,
and the pure-NumPy CNN layers with forward/backward passes verified against
float64 finite-difference gradient checks. The training loop and experiments
belong to Phase 4 (see the roadmap).

---

## Table of contents

1. [Assignment objective](#1-assignment-objective)
2. [The two implementations](#2-the-two-implementations)
3. [Dataset](#3-dataset)
4. [Dataset preprocessing](#4-dataset-preprocessing)
5. [Project structure](#5-project-structure)
6. [Environment setup](#6-environment-setup)
7. [Dependency installation](#7-dependency-installation)
8. [Dataset preparation](#8-dataset-preparation)
9. [Testing](#9-testing)
10. [Expected dataset outputs](#10-expected-dataset-outputs)
11. [Development roadmap](#11-development-roadmap)
12. [Implementation status](#12-implementation-status)

---

## 1. Assignment objective

Implement the same convolutional neural network twice and compare the two
implementations:

1. **From scratch**, using only Python and NumPy: convolution, ReLU, max
   pooling, dense layers, softmax, cross-entropy loss, forward propagation,
   backpropagation and stochastic gradient descent written by hand.
2. **With Keras/TensorFlow**, using the framework's layer and autodiff support.

The comparison covers classification accuracy, training behaviour and the
engineering effort each approach requires. A hand-written implementation shows
how the mathematics of a CNN actually works; the framework implementation shows
what production tooling hides. Reproducing the same results from both is the
evidence that the manual implementation is correct.

## 2. The two implementations

| Task | Scratch CNN (`NumPy`) | Keras CNN (`TensorFlow`) |
|---|---|---|
| Loading the processed dataset | `numpy.load` | `numpy.load` |
| Convolution | manual `Conv2D.forward/backward` | `keras.layers.Conv2D` |
| Activation | manual ReLU | `activation="relu"` |
| Pooling | manual max pooling with argmax routing | `keras.layers.MaxPooling2D` |
| Fully connected | manual `Dense.forward/backward` | `keras.layers.Dense` |
| Output | manual stable softmax | `activation="softmax"` |
| Loss | manual categorical cross-entropy | `categorical_crossentropy` |
| Gradients | derived and coded by hand | automatic differentiation |
| Weight updates | manual SGD loop | `optimizer=SGD` |

### Where TensorFlow is and is not used

- **Allowed:** downloading Fashion-MNIST
  (`keras.datasets.fashion_mnist.load_data()`), and the separate Keras reference
  model.
- **Not allowed:** any computation inside the scratch model. TensorFlow/Keras
  does not perform the scratch CNN's forward propagation, backpropagation,
  gradient calculation or weight updates.

`src/cnn_scratch/data/prepare_data.py` imports Keras lazily inside
`load_fashion_mnist()` only, and nothing in the dataset module imports TensorFlow
at module level. Both implementations consume the same processed `.npz` files, so
their inputs are identical by construction.

### The frozen architecture

Both implementations build exactly the network below. It is declared in
`configs/experiment_config.json`, and `src/cnn_scratch/models/architecture.py`
derives the shapes and parameter counts from that declaration, so the
documentation cannot drift from the configuration.

| Layer | Configuration | Output (one sample) | Trainable parameters |
|---|---|---|---|
| Input | 28 x 28 grayscale, NHWC | `(28, 28, 1)` | 0 |
| Conv2D | 8 filters, 3x3, stride 1, valid, bias | `(26, 26, 8)` | 80 |
| ReLU | element-wise | `(26, 26, 8)` | 0 |
| MaxPool2D | 2x2, stride 2, valid | `(13, 13, 8)` | 0 |
| Flatten | row-major | `(1352,)` | 0 |
| Dense | 10 units, bias, logits | `(10,)` | 13,530 |
| Softmax | 10 class probabilities | `(10,)` | 0 |

Total trainable parameters: **13,610**. Batch shapes prepend the batch
dimension — `(32, 26, 26, 8)`, `(32, 1352)`, `(32, 10)` with the configured batch
size.

Initial settings shared by both models: seed 42, NHWC layout, He normal
initialisation, zero biases, SGD with learning rate 0.01, batch size 32, 5
epochs, mean categorical cross-entropy.

Derivations with worked numerical examples:
[`docs/cnn_mathematical_foundations.md`](docs/cnn_mathematical_foundations.md).
One training step traced backwards, with every cached tensor and gradient shape:
[`docs/backpropagation_walkthrough.md`](docs/backpropagation_walkthrough.md).

## 3. Dataset

[Fashion-MNIST](https://github.com/zalandoresearch/fashion-mnist) — 70,000
28x28 grayscale clothing images from Zalando, 10 classes.

| Property | Value |
|---|---|
| Official training split | 60,000 images |
| Official test split | 10,000 images |
| Image size | 28 x 28 grayscale |
| Classes | 10 |

| Label | Class | Label | Class |
|---|---|---|---|
| 0 | T-shirt/top | 5 | Sandal |
| 1 | Trouser | 6 | Shirt |
| 2 | Pullover | 7 | Sneaker |
| 3 | Dress | 8 | Bag |
| 4 | Coat | 9 | Ankle boot |

**Experimental subsets used here** (fixed seed 42, all ten classes,
never mixing the two official splits):

| Split | Per class | Total | Drawn from |
|---|---|---|---|
| Training | 1,000 | 10,000 | official training split only |
| Test | 200 | 2,000 | official test split only |

The test subset is held out for final evaluation and must not be used for
training or hyperparameter tuning.

## 4. Dataset preprocessing

- Pixel values cast to `float32` and divided by 255 → range `[0.0, 1.0]`.
- Channel axis appended: `(n, 28, 28)` → `(n, 28, 28, 1)`.
- Labels kept as integer class IDs; one-hot encoding only where a specific
  implementation requires it.
- Balanced sampling per class without replacement, then a seeded permutation.

Resulting arrays:

```
x_train  (10000, 28, 28, 1)  float32   y_train  (10000,)  uint8
x_test   (2000, 28, 28, 1)   float32   y_test   (2000,)   uint8
```

Full specification: [`docs/dataset_documentation.md`](docs/dataset_documentation.md).

## 5. Project structure

```text
cnn-from-scratch/
├── README.md
├── requirements.txt
├── .gitignore
├── .gitattributes
├── LICENSE
├── conftest.py                   # puts src/ on sys.path for pytest
├── configs/
│   └── experiment_config.json    # seed, subset sizes, initial training settings
├── data/
│   ├── raw/                      # acquisition cache (git-ignored)
│   ├── processed/                # generated subsets (git-ignored)
│   └── README.md
├── notebooks/
│   └── README.md
├── src/
│   └── cnn_scratch/
│       ├── __init__.py
│       ├── data/
│       │   ├── __init__.py
│       │   ├── prepare_data.py   # pipeline + CLI
│       │   └── dataset_utils.py  # constants, validation, NPZ/JSON I/O
│       ├── models/
│       │   ├── __init__.py
│       │   └── architecture.py   # frozen shapes and parameter counts
│       └── utils/
│           ├── __init__.py
│           └── visualization.py  # sample grid, class distribution
├── tests/
│   ├── __init__.py
│   ├── test_data_preparation.py
│   └── test_architecture.py
├── results/
│   ├── figures/
│   ├── metrics/
│   └── README.md
├── docs/
│   ├── project_architecture.md
│   ├── dataset_documentation.md
│   ├── cnn_mathematical_foundations.md
│   ├── backpropagation_walkthrough.md
│   └── development_roadmap.md
└── report/
    └── README.md
```

## 6. Environment setup

Tested on Windows 11 with PowerShell. Python **3.13.5** (64-bit) is used because
TensorFlow 2.21.0 publishes a native `cp313-win_amd64` wheel; Python 3.10-3.13
are all supported by that release. Older TensorFlow versions stop at Python
3.12, so check the wheel tags before changing versions.

The global Python installation is never modified: everything happens inside
`.venv/`.

```powershell
# From the repository root
cd path\to\cnn-from-scratch

# 1. Confirm the interpreter
py -3.13 --version

# 2. Create the isolated environment
py -3.13 -m venv .venv

# 3. Activate it (PowerShell)
.\.venv\Scripts\Activate.ps1

# If activation is blocked by the execution policy, allow it for this session:
#   Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass

# 4. Upgrade the installer
python -m pip install --upgrade pip

# 5. Install the pinned dependencies
pip install -r requirements.txt
```

`python` now resolves to `.venv\Scripts\python.exe`. Verify with
`python -c "import sys; print(sys.executable)"`.

On Git Bash or cmd the activation differs (`.venv/Scripts/activate` or
`.venv\Scripts\activate.bat`); installing from the full path
`.venv\Scripts\python.exe -m pip ...` also works without activation.

### Jupyter kernel

Register a dedicated kernel so notebooks use the project environment:

```powershell
python -m ipykernel install --user --name cnn-from-scratch --display-name "Python (cnn-from-scratch)"
```

Select **Python (cnn-from-scratch)** in VS Code or Jupyter.

## 7. Dependency installation

Pinned versions in `requirements.txt`, installed and verified during Phase 1:

| Package | Role |
|---|---|
| `numpy` | array math, dataset storage, the scratch CNN |
| `tensorflow` | Keras layers, autodiff, reference model |
| `matplotlib` | dataset and training figures |
| `scikit-learn` | metrics and confusion matrices (later phases) |
| `jupyter`, `ipykernel` | notebooks |
| `pytest` | automated tests |

Check the environment in one line:

```powershell
python -c "import numpy, tensorflow, matplotlib, sklearn, pytest; print(numpy.__version__, tensorflow.__version__)"
```

> TensorFlow on native Windows is CPU-only; training runs on CPU, which is one
> reason for the reduced 10,000-image training subset.

## 8. Dataset preparation

```powershell
python -m cnn_scratch.data.prepare_data
```

The package is not installed, so `src/` must be importable. With `.venv`
activated, set the path once for the session:

```powershell
$env:PYTHONPATH = "src"
python -m cnn_scratch.data.prepare_data
```

Or bypass activation entirely:

```powershell
$env:PYTHONPATH = "src"; .\.venv\Scripts\python.exe -m cnn_scratch.data.prepare_data
```

The command downloads Fashion-MNIST on first run (cached by Keras outside the
repository), selects the balanced subsets, normalises them, validates them and
writes the artefacts below. It also saves two figures.

Useful flags:

```powershell
python -m cnn_scratch.data.prepare_data --help
python -m cnn_scratch.data.prepare_data --no-figure
python -m cnn_scratch.data.prepare_data --output data/processed/other_name.npz
```

Settings come from `configs/experiment_config.json` (`random_seed`,
`train_samples_per_class`, `test_samples_per_class`, ...). Output is
deterministic, so re-running with the same configuration reproduces the same
arrays.

## 9. Testing

```powershell
pytest -v
```

The suite checks the real processed dataset and, if it is missing, rebuilds it
through the project pipeline first:

- training shape `(10000, 28, 28, 1)`, test shape `(2000, 28, 28, 1)`;
- image dimensions and channel axis;
- `float32` dtype and pixel range `[0, 1]`;
- integer labels restricted to class IDs 0-9;
- exactly 1,000 training and 200 test examples per class;
- determinism: the same seed reproduces identical arrays, a different seed does
  not;
- save/reload round-trip preserves values and dtypes;
- validation rejects wrong shapes, unnormalised pixels and unbalanced splits;
- metadata consistency with the saved arrays;
- the frozen architecture: each layer's output shape, the 1352-wide flatten,
  convolution parameters 80, dense parameters 13,530, total 13,610,
  configuration consistency, and rejection of invalid layer specifications.
- every NumPy layer's forward and backward pass: shapes, forward values,
  analytic gradients versus float64 central differences (convolution input,
  weights and bias; dense input, weights and bias; pooling input; the general
  softmax vector-Jacobian product; the fused softmax-cross-entropy logits
  gradient), pooling tie handling, overlapping-window gradient accumulation,
  and the full `ScratchCNN` forward/backward integration.

## 10. Expected dataset outputs

After preparation:

| Path | Content |
|---|---|
| `data/processed/fashion_mnist_subset.npz` | `x_train`, `y_train`, `x_test`, `y_test` |
| `data/processed/fashion_mnist_subset_metadata.json` | seed, source, sizes, class distributions, preprocessing |
| `results/figures/dataset_samples.png` | 10 x 10 sample grid, one row per class |
| `results/figures/class_distribution.png` | per-class counts for both splits |

Console summary:

```
Fashion-MNIST subsets prepared
  train: (10000, 28, 28, 1) float32
  test : (2000, 28, 28, 1) float32
  train class counts: {0: 1000, 1: 1000, ...}
  test  class counts: {0: 200, 1: 200, ...}
```

The dataset files are git-ignored and must be regenerated by anyone cloning the
repository; the command is deterministic.

## 11. Development roadmap

| Phase | Scope | Status |
|---|---|---|
| 1 | Repository foundation, environment, dataset pipeline, docs, tests | **complete, published** |
| 2 | Mathematical foundations, frozen architecture, shape and parameter validation | **complete** |
| 3 | NumPy layers (conv, ReLU, pool, dense, softmax) + gradient checks | **complete** |
| 4 | Scratch CNN: forward/backward, SGD training loop | not started |
| 5 | Equivalent Keras model, identical settings | not started |
| 6 | Comparison: accuracy, loss curves, timing, confusion matrices | not started |
| 7 | Report finalisation | not started |

Details and per-phase checklists:
[`docs/development_roadmap.md`](docs/development_roadmap.md).

## 12. Implementation status

**Implemented and verified**

- Repository structure, Git configuration, licensing, documentation.
- Isolated `.venv` with pinned, verified dependencies.
- Fashion-MNIST acquisition, balanced subset selection, preprocessing,
  validation, persistence and metadata.
- Visualisation utilities.
- Mathematical foundations and a backpropagation walkthrough (`docs/`), with
  worked numerical examples.
- The frozen architecture with validated output shapes and parameter counts
  (`src/cnn_scratch/models/architecture.py`).
- Automated pytest suites for the dataset layer and the architecture.
- Pure-NumPy `Conv2D`, `ReLU`, `MaxPool2D`, `Flatten`, `Dense`, stable softmax
  and categorical cross-entropy layers with manual forward and backward passes
  (`src/cnn_scratch/models/layers.py`), and the assembled `ScratchCNN`
  forward/backward pass (`src/cnn_scratch/models/scratch_cnn.py`).
- Gradient validation: analytic gradients match float64 central differences
  for every parameter tensor, for the general softmax VJP and for the fused
  `(P - Y) / B` logits gradient; 77 tests pass.

**Not implemented yet**

- Gradient descent and any training loop.
- The Keras model.
- Any accuracy or loss number.

No training has been run and no model metric exists in this repository. Results
will only be added after they are measured.

## References

- Xiao, Rasul & Vollgraf, *Fashion-MNIST: a Novel Image Dataset for Benchmarking
  Machine Learning Algorithms*, 2017.
- LeCun, Bottou, Bengio & Haffner, *Gradient-Based Learning Applied to Document
  Recognition*, 1998.
- TensorFlow/Keras documentation: <https://www.tensorflow.org/>

## Licence

MIT — see [`LICENSE`](LICENSE).
