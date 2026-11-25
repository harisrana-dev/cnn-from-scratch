# Development Roadmap

Work proceeds in phases. Each phase ends with a working, committed state.
Statuses below reflect what actually exists in the repository, not intentions.

## Phase 1 — Repository foundation, environment, dataset (complete)

- [x] Inspect Python versions and confirm TensorFlow compatibility on Windows
- [x] Isolated `.venv` with pinned dependencies
- [x] Repository structure, `.gitignore`, `.gitattributes`, `LICENSE`
- [x] Balanced Fashion-MNIST subsets (1,000/class train, 200/class test, seed 42)
- [x] Preprocessing: float32, [0, 1], channel axis, integer labels
- [x] Processed `.npz` plus metadata JSON
- [x] Dataset visualisation utilities
- [x] Documentation (README, architecture, dataset, roadmap)
- [x] pytest suite for dataset preparation

**Not in Phase 1:** no layers, no losses, no gradients, no training.

## Phase 2 — Mathematical foundations and architecture design

- [ ] Written derivation of convolution, pooling and dense forward passes
- [ ] Backpropagation derivations for each layer type
- [ ] Cross-entropy and softmax gradient
- [ ] Fix the architecture shared by both implementations, e.g.
      `Conv(8, 3x3, padding='same') → ReLU → MaxPool(2x2) → Conv(16, 3x3) → ReLU
      → MaxPool(2x2) → Flatten → Dense(64) → ReLU → Dense(10) → Softmax`
- [ ] Shape-check table for every layer transition
- [ ] Record the design in `docs/`

## Phase 3 — NumPy layer implementations

- [ ] `Conv2D` forward (im2col or direct) and backward (dW, db, dX)
- [ ] `ReLU` forward/backward
- [ ] `MaxPool2D` forward/backward with argmax routing
- [ ] `Dense` forward/backward
- [ ] `Softmax` + categorical cross-entropy, numerically stable
- [ ] Gradient checking against finite differences
- [ ] Unit tests for each layer (analytic gradient vs. numerical)

## Phase 4 — Scratch CNN model and training

- [ ] He/Xavier initialisation with the project seed
- [ ] Forward and backward passes over a mini-batch
- [ ] SGD update (learning rate 0.01, batch size 32 as a starting point)
- [ ] Mini-batch iterator, loss/accuracy history
- [ ] Determinism check: identical seeds reproduce identical losses
- [ ] Export history to `results/metrics/`

## Phase 5 — Keras/TensorFlow reference CNN

- [ ] Equivalent architecture in Keras
- [ ] Same optimiser, loss, batch size and epoch count
- [ ] Training history export to `results/metrics/`
- [ ] Assert both models consume the identical `.npz` arrays

## Phase 6 — Comparison and analysis

- [ ] Accuracy, loss curves, training time per epoch
- [ ] Confusion matrices for both models
- [ ] Error analysis on misclassified samples
- [ ] Discussion of accuracy/speed/complexity trade-offs
- [ ] Fill `report/` with measured results only

## Phase 7 — Finalisation

- [ ] README updated with measured results
- [ ] Reproduction instructions verified from a clean environment
- [ ] Submission package prepared

## Conventions for every phase

1. Commit at each milestone with a descriptive message.
2. Tests before the phase is declared complete.
3. No metric appears in documentation before it has been measured.
4. Configuration changes are recorded in `configs/experiment_config.json` and
   noted here.
