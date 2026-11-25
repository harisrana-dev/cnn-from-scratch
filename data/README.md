# Data

This directory holds the Fashion-MNIST data. **No dataset file is committed to
Git**; everything here except this README and the `.gitkeep` placeholders is
ignored.

```
data/
├── raw/         # untouched downloads / Keras cache (git-ignored)
└── processed/   # generated subsets used by both implementations (git-ignored)
```

## Sources

| Split | Source | Size |
|---|---|---|
| Training | official Fashion-MNIST training split | 60,000 images |
| Test | official Fashion-MNIST test split | 10,000 images |
| Subset used here | balanced sample of the two official splits | 10,000 train / 2,000 test |

- Official repository: <https://github.com/zalandoresearch/fashion-mnist>
- Acquisition in this project: `keras.datasets.fashion_mnist.load_data()`, used
  **only** to download data. The scratch CNN performs no framework computations.
- The Keras cache lives outside the project, at
  `%USERPROFILE%\.keras\datasets\fashion-mnist\`, so `data/raw/` stays empty
  unless you copy the archives there yourself. It is a git-ignored scratch
  location, not a required input.

## Generating the processed dataset

From the repository root, with the virtual environment activated:

```powershell
python -m cnn_scratch.data.prepare_data
```

This writes:

| File | Contents |
|---|---|
| `data/processed/fashion_mnist_subset.npz` | arrays `x_train`, `y_train`, `x_test`, `y_test` |
| `data/processed/fashion_mnist_subset_metadata.json` | seed, class distribution, source, preprocessing, subset sizes |

Shapes and dtypes:

```
x_train  (10000, 28, 28, 1)  float32   values in [0, 1]
y_train  (10000,)            uint8     class IDs 0-9
x_test   (2000, 28, 28, 1)   float32   values in [0, 1]
y_test   (2000,)             uint8     class IDs 0-9
```

The dataset is deterministic: the same configuration and seed reproduce
identical arrays. Delete the generated files at any time and re-run the command
to rebuild them.

## Rules

- Training samples come only from the official training split, test samples only
  from the official test split. They are never mixed.
- The test split is held out for final evaluation; it must not be used for
  training or hyperparameter tuning.
- Both implementations load these exact files, so their inputs are identical.

See `docs/dataset_documentation.md` for the full specification.
