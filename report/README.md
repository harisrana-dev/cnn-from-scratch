# Report

Materials for the written assignment submission.

| Item | Status |
|---|---|
| Report document (`.docx`/`.pdf`) | not created yet |
| Figures sourced from `results/figures/` | dataset figures available after Phase 1 |
| Comparison tables | pending Phase 5-6, once measurements exist |

## Suggested structure

1. **Introduction** – aim, comparison between a manual NumPy CNN and a
   Keras/TensorFlow CNN.
2. **Dataset** – Fashion-MNIST, balanced subsets, preprocessing
   (see `docs/dataset_documentation.md`).
3. **Methodology**
   - Convolution, activation, pooling, fully connected layers.
   - Forward propagation and backpropagation, derived by hand.
   - Gradient descent and the SGD update rule.
4. **Implementation** – code walkthrough of the scratch model and the Keras
   equivalent.
5. **Experiments and results** – losses, accuracies, confusion matrices,
   training time (filled in from `results/metrics/` only).
6. **Discussion** – accuracy differences, training speed, implementation
   complexity, numerical stability.
7. **Conclusion and future work**.

Large report binaries are not tracked by default; add the final PDF explicitly
if required by the submission.
