# Dataset Documentation

## Source

| Property | Value |
|---|---|
| Name | Fashion-MNIST |
| Repository | <https://github.com/zalandoresearch/fashion-mnist> |
| Paper | Xiao, Rasul & Vollgraf (2017), *Fashion-MNIST: a Novel Image Dataset for Benchmarking Machine Learning Algorithms* |
| Licence | MIT (Zalando SE) |
| Official training split | 60,000 images |
| Official test split | 10,000 images |
| Image format | 28 x 28 grayscale, 8-bit, pixel values 0-255 |
| Classes | 10 |

Fashion-MNIST is a drop-in replacement for MNIST: same image size and split
sizes, but clothing items instead of handwritten digits, which makes it
noticeably harder for small models.

## Classes

| Label | Class |
|---|---|
| 0 | T-shirt/top |
| 1 | Trouser |
| 2 | Pullover |
| 3 | Dress |
| 4 | Coat |
| 5 | Sandal |
| 6 | Shirt |
| 7 | Sneaker |
| 8 | Bag |
| 9 | Ankle boot |

## Experimental subsets

All ten classes are used. Two balanced subsets are drawn once and then reused by
both implementations.

| Split | Images per class | Total | Drawn from |
|---|---|---|---|
| Training | 1,000 | 10,000 | official 60,000 training images |
| Test | 200 | 2,000 | official 10,000 test images |

- A fixed random seed of **42** drives the selection
  (`numpy.random.default_rng(42)`); the test subset uses `42 + 1` so the two
  splits are not drawn from the same index sequence inside their respective
  source arrays.
- Sampling is per class **without replacement**, followed by a seeded
  permutation, so no image is duplicated and classes are not stored in blocks.
- The official training and test splits are never mixed.
- The test subset is held out: no training or hyperparameter tuning may use it.

Why a subset? A NumPy convolution implementation is far slower than cuDNN; 10,000
training images keep a full epoch measurable in minutes on CPU while retaining
every class.

## Preprocessing

| Step | Detail |
|---|---|
| Cast | `uint8` → `float32` |
| Normalisation | divide by 255 → values in `[0.0, 1.0]` |
| Channel axis | appended, `(n, 28, 28)` → `(n, 28, 28, 1)` |
| Labels | remain integer class IDs (`uint8`); one-hot only where an implementation needs it |

Final shapes:

```
x_train  (10000, 28, 28, 1)  float32
y_train  (10000,)            uint8
x_test   (2000, 28, 28, 1)   float32
y_test   (2000,)             uint8
```

## Artefacts

| File | Description |
|---|---|
| `data/processed/fashion_mnist_subset.npz` | compressed archive with `x_train`, `y_train`, `x_test`, `y_test` |
| `data/processed/fashion_mnist_subset_metadata.json` | seed, source, subset sizes, class distributions, preprocessing record |
| `results/figures/dataset_samples.png` | sample grid, one row per class |
| `results/figures/class_distribution.png` | per-class counts for both splits |

Both data files are git-ignored; regenerate them with:

```powershell
python -m cnn_scratch.data.prepare_data
```

Optional flags: `--config`, `--output`, `--metadata`, `--no-figure`.

## Metadata example

```json
{
  "dataset": "Fashion-MNIST",
  "random_seed": 42,
  "num_classes": 10,
  "image_shape": [28, 28, 1],
  "subset_sizes": {
    "train_samples_per_class": 1000,
    "test_samples_per_class": 200,
    "train_samples_total": 10000,
    "test_samples_total": 2000
  },
  "preprocessing": {
    "dtype": "float32",
    "normalization": "divide_by_255",
    "output_pixel_range": [0.0, 1.0],
    "channel_dimension": "added as trailing axis (n, 28, 28, 1)",
    "labels": "integer class IDs, no one-hot encoding"
  }
}
```

(Abridged; the generated file also records per-split shapes, class
distributions and a UTC generation timestamp.)

## Validation guarantees

`validate_dataset()` fails loudly if any of the following is wrong, and the test
suite asserts the same properties on the real artefacts:

- image tensor is 4-D with shape `(n, 28, 28, 1)`;
- sample count equals `samples_per_class * 10`;
- images are floating point with values in `[0, 1]`;
- labels are integers in `[0, 9]`;
- every class appears exactly `samples_per_class` times.

## Intended use

| Purpose | Allowed |
|---|---|
| Training both CNNs on `x_train`/`y_train` | yes |
| Final evaluation on `x_test`/`y_test` | yes |
| Choosing architecture or hyperparameters on `x_test`/`y_test` | no — use a validation split of `x_train` |

## Provenance and reproducibility

The pipeline is deterministic given the configuration in
`configs/experiment_config.json`. Two runs with the same seed produce
bit-identical arrays; `tests/test_data_preparation.py` verifies this by rebuilding
the dataset and comparing it with the saved artefacts.

Keras is used **only** to download the raw data. All preprocessing after the
download is plain NumPy so the scratch model never depends on framework
behaviour.
