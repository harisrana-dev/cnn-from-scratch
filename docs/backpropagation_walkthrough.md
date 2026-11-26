# Backpropagation Walkthrough

This document traces **one complete training step** backwards through the frozen
architecture, stage by stage, with the real dimensions. It is the specification
Phase 3 implements against and the explanation to reproduce in the viva.

**Scope.** Mathematics and shape bookkeeping only. No backward pass, loss,
optimiser or training loop has been implemented yet, and no result is quoted.

Notation and derivations are in
[`cnn_mathematical_foundations.md`](cnn_mathematical_foundations.md); this
document applies them. Architecture and parameter counts are declared in
`configs/experiment_config.json` and computed by
`src/cnn_scratch/models/architecture.py`, so the numbers below are not copied by
hand.

**Frozen architecture ($B = 32$):**

```text
X        (32, 28, 28, 1)
  conv 8 filters 3x3, valid, bias      -> Z1  (32, 26, 26, 8)
  ReLU                                  -> A1  (32, 26, 26, 8)
  maxpool 2x2 stride 2                  -> A2  (32, 13, 13, 8)
  flatten                               -> A3  (32, 1352)
  dense 10 units, bias (logits)         -> Z3  (32, 10)
  softmax                               -> P   (32, 10)
  mean categorical cross-entropy        -> L   scalar
```

Total trainable parameters: 80 + 13,530 = **13,610**.

---

## 1. One training step end to end

```text
forward   X -> Z1 -> A1 -> A2 -> A3 -> Z3 -> P -> L        (cache what backward needs)
backward  L -> G  -> dW2, db2, dA3 -> dA2 -> dA1 -> dZ1 -> dW1, db1, dX
update    W1 -= lr*dW1   b1 -= lr*db1   W2 -= lr*dW2   b2 -= lr*db2
```

Backpropagation is the chain rule applied in reverse topological order: the
gradient of the loss with respect to a layer's output is computed first, then
converted into gradients for that layer's parameters and for its input, which
becomes the "incoming gradient" of the previous layer.

The task list for one step, in order:

| # | Stage | Computes | Feeds |
|---|---|---|---|
| 1 | Loss + softmax | $\mathbf{G} = (\mathbf{P}-\mathbf{Y})/B$ | dense |
| 2 | Dense | $\nabla_{\mathbf{W}_2}$, $\nabla_{\boldsymbol{\beta}_2}$, $\nabla_{\mathbf{A}_3}$ | flatten |
| 3 | Flatten | $\nabla_{\mathbf{A}_2}$ (reshape) | max pool |
| 4 | Max pool | $\nabla_{\mathbf{A}_1}$ (scatter to argmax) | ReLU |
| 5 | ReLU | $\nabla_{\mathbf{Z}_1}$ (mask) | convolution |
| 6 | Convolution | $\nabla_{\mathbf{W}_1}$, $\nabla_{\boldsymbol{\beta}_1}$, $\nabla_{\mathbf{X}}$ | input / gradient check |

No layer computes its own gradient before the layer after it has supplied one:
this is why the pass runs right to left.

---

## 2. What the forward pass must cache

Backward passes need values that no longer exist once the forward pass finishes.
The complete cache for one mini-batch of 32:

| Tensor | Shape | dtype / bytes | Needed by | Why |
|---|---|---|---|---|
| $\mathbf{X}$ | $(32,28,28,1)$ | float32, 100,352 elements | conv backward | $\nabla_{\mathbf{W}_1}$ is a weighted sum of inputs |
| $\mathbf{W}_1$ | $(3,3,1,8)$ | float32, 72 | conv backward | $\nabla_{\mathbf{X}}$ is a weighted sum of filters |
| $\mathbf{Z}_1$ (or its mask) | $(32,26,26,8)$ | bool, 173,056 | ReLU backward | which activations passed gradient |
| argmax of each pool window | $(32,13,13,8)$ | int, 43,264 | pool backward | where to route the gradient |
| input shape to flatten | $(32,13,13,8)$ | — | flatten backward | reshape target |
| $\mathbf{A}_3$ | $(32,1352)$ | float32, 43,264 | dense backward | $\nabla_{\mathbf{W}_2} = \mathbf{A}_3^{\top}\mathbf{G}$ |
| $\mathbf{W}_2$ | $(1352,10)$ | float32, 13,520 | dense backward | $\nabla_{\mathbf{A}_3} = \mathbf{G}\mathbf{W}_2^{\top}$ |
| $\mathbf{Z}_3$ | $(32,10)$ | float32, 320 | softmax | probabilities |
| $\mathbf{P}$ | $(32,10)$ | float32, 320 | loss gradient | $\mathbf{G} = (\mathbf{P}-\mathbf{Y})/B$ |
| labels $y$ | $(32,)$ | uint8/int | loss | one-hot target, or direct indexing |

The activations dominate the cache (173,056 float32 values ≈ 692 KB for the
convolution output). ReLU's cache can be stored as a boolean mask, which costs
one byte per element instead of four and is all the backward pass reads.

$\mathbf{A}_1$ (the ReLU output) is **not** cached: max pooling needs only the
argmax, which is recorded from $\mathbf{A}_1$ during the forward pass. $\mathbf{A}_2$
is not needed either, only its shape.

---

## 3. Stage-by-stage backward trace

### Stage 1 — Cross-entropy loss and the logits gradient

**Forward.** $\mathbf{P} = \operatorname{softmax}(\mathbf{Z}_3)$ with the
row-max shift, then
$L = -\frac{1}{B}\sum_{b}\log \mathbf{P}_{b,y_b}$.

**Cached.** $\mathbf{P}$ (and the integer labels).

**Incoming gradient.** None — the pass starts from the scalar $L$, whose gradient
with respect to itself is 1.

**Backward operation.**
$\;\mathbf{G} = \dfrac{\partial L}{\partial \mathbf{Z}_3} = \dfrac{1}{B}(\mathbf{P} - \mathbf{Y})$.

$\mathbf{Y}$ is the one-hot expansion of the integer labels. The softmax
Jacobian is never formed: the combined derivative of
[Section 14](cnn_mathematical_foundations.md#14-combined-softmax-cross-entropy-derivative)
collapses to a subtraction.

**Incoming shape.** — (scalar seed)
**Outgoing shape.** $(32, 10)$, the same as $\mathbf{Z}_3$.

**Parameter gradients.** None: softmax has no parameters.
**Propagated.** $\mathbf{G}$ to the dense layer.

**Checks.** Each row of $\mathbf{G}$ sums to 0; entries lie in $[-1/B, 1/B]$; the
true class entry is negative.

---

### Stage 2 — Dense layer

**Forward.** $\mathbf{Z}_3 = \mathbf{A}_3\mathbf{W}_2 + \mathbf{b}_2$ with
$\mathbf{A}_3 \in \mathbb{R}^{32\times1352}$, $\mathbf{W}_2 \in
\mathbb{R}^{1352\times10}$, $\mathbf{b}_2 \in \mathbb{R}^{10}$.

**Cached.** $\mathbf{A}_3$, $\mathbf{W}_2$.

**Incoming gradient.** $\mathbf{G} = \partial L/\partial\mathbf{Z}_3$,
shape $(32, 10)$.

**Backward operation.**

$$
\nabla_{\mathbf{W}_2} L = \mathbf{A}_3^{\top}\mathbf{G}
\qquad
\nabla_{\mathbf{b}_2} L = \sum_{b=0}^{31}\mathbf{G}_{b,:}
\qquad
\nabla_{\mathbf{A}_3} L = \mathbf{G}\,\mathbf{W}_2^{\top}
$$

**Outgoing shape.** $\nabla_{\mathbf{W}_2}$: $(1352, 10)$ = 13,520 values;
$\nabla_{\mathbf{b}_2}$: $(10,)$; $\nabla_{\mathbf{A}_3}$: $(32, 1352)$.

**Parameter gradients.** 13,530 in total (weights plus bias), each the same shape
as the parameter it updates.

**Propagated.** $\nabla_{\mathbf{A}_3}$ to the flatten layer.

**Note on the batch dimension.** It is summed over **only** in the parameter
gradients (through $\mathbf{A}_3^{\top}$ and the column sum). The gradient with
respect to the input never mixes samples, which is why each row of
$\nabla_{\mathbf{A}_3}$ can be reshaped independently in the next stage.

---

### Stage 3 — Flatten

**Forward.** $\mathbf{A}_3 = \operatorname{reshape}(\mathbf{A}_2, (32, 1352))$,
row-major: $\mathbf{A}_{3,b,(i\cdot13+j)\cdot8+f} = \mathbf{A}_{2,b,i,j,f}$.

**Cached.** The input shape $(32, 13, 13, 8)$.

**Incoming gradient.** $\nabla_{\mathbf{A}_3}$, shape $(32, 1352)$.

**Backward operation.** The inverse reshape:
$\nabla_{\mathbf{A}_2} = \operatorname{reshape}(\nabla_{\mathbf{A}_3}, (32,13,13,8))$.

**Outgoing shape.** $(32, 13, 13, 8)$.

**Parameter gradients.** None.
**Propagated.** $\nabla_{\mathbf{A}_2}$ to max pooling, element-wise unchanged.

---

### Stage 4 — Max pooling

**Forward.** For each channel $f$ and each non-overlapping $2\times2$ window,
$\mathbf{A}_{2,b,i,j,f} = \max_{u,v\in\{0,1\}} \mathbf{A}_{1,b,2i+u,2j+v,f}$,
recording
$(\hat u,\hat v)_{b,i,j,f} = \argmax$ (first maximum in row-major order — see
[Section 7](cnn_mathematical_foundations.md#7-max-pooling-forward) for the
tie-breaking rule).

**Cached.** The argmax positions, shape $(32, 13, 13, 8)$ — one index per output
value, 43,264 entries.

**Incoming gradient.** $\nabla_{\mathbf{A}_2}$, shape $(32, 13, 13, 8)$.

**Backward operation.** Start from a zero tensor
$\mathbf{0} \in \mathbb{R}^{32\times26\times26\times8}$ and add each output
gradient at its recorded position:

$$
\nabla_{\mathbf{A}_1}\!\left[b,\,2i+\hat u,\,2j+\hat v,\,f\right]
\;\mathrel{+}=\; \nabla_{\mathbf{A}_2}[b,i,j,f].
$$

The remaining three positions of every window stay at zero.

**Outgoing shape.** $(32, 26, 26, 8)$ — the shape the ReLU layer produced.

**Parameter gradients.** None. The gradient is routed, never scaled: with
non-overlapping windows no position receives two contributions, so `+=` and `=`
coincide here (the `+=` form is used for safety and for overlapping variants).

**Propagated.** $\nabla_{\mathbf{A}_1}$ to ReLU.

---

### Stage 5 — ReLU

**Forward.** $\mathbf{A}_1 = \max(0, \mathbf{Z}_1)$ element-wise on
$(32,26,26,8)$.

**Cached.** The pre-activation $\mathbf{Z}_1$ (or the mask
$\mathbf{Z}_1 > 0$), shape $(32,26,26,8)$.

**Incoming gradient.** $\nabla_{\mathbf{A}_1}$, shape $(32, 26, 26, 8)$.

**Backward operation.**

$$
\nabla_{\mathbf{Z}_1} = \nabla_{\mathbf{A}_1} \odot \mathbb{1}[\mathbf{Z}_1 > 0].
$$

**Outgoing shape.** $(32, 26, 26, 8)$, unchanged.

**Parameter gradients.** None.
**Propagated.** $\nabla_{\mathbf{Z}_1}$ to the convolution.

**Check.** All entries of $\nabla_{\mathbf{Z}_1}$ must be zero wherever
$\mathbf{Z}_1 \le 0$; if the mask was taken from $\mathbf{A}_1$ instead, values
in the negative region are wrong (they look right only where $\mathbf{A}_1 = 0$
coincides with a zero gradient).

---

### Stage 6 — Convolution

**Forward.** For each filter $f$ and each of the $26\times26$ valid positions:

$$
\mathbf{Z}_{1,b,i,j,f} = \beta_{1,f} + \sum_{c=0}^{0}\sum_{u=0}^{2}\sum_{v=0}^{2}
\mathbf{X}_{b,\,i+u,\,j+v,\,c}\;\mathbf{W}_{1,u,v,c,f}.
$$

**Cached.** The input $\mathbf{X}$ $(32,28,28,1)$ and the filters
$\mathbf{W}_1$ $(3,3,1,8)$.

**Incoming gradient.** $\nabla_{\mathbf{Z}_1} = \partial L/\partial \mathbf{Z}_1$,
shape $(32, 26, 26, 8)$.

**Backward operation.** Three gradients, all from the same cached pair:

$$
\frac{\partial L}{\partial \mathbf{W}_{1,u,v,c,f}}
= \sum_{b=0}^{31}\sum_{i=0}^{25}\sum_{j=0}^{25}
\frac{\partial L}{\partial \mathbf{Z}_{1,b,i,j,f}}\;
\mathbf{X}_{b,\,i+u,\,j+v,\,c},
$$

$$
\frac{\partial L}{\partial \beta_{1,f}}
= \sum_{b=0}^{31}\sum_{i=0}^{25}\sum_{j=0}^{25}
\frac{\partial L}{\partial \mathbf{Z}_{1,b,i,j,f}},
$$

$$
\frac{\partial L}{\partial \mathbf{X}_{b,\,h,\,w,\,c}}
= \sum_{f=0}^{7}\;\sum_{\substack{(i,j,u,v)\\ i+u=h,\; j+v=w}}
\frac{\partial L}{\partial \mathbf{Z}_{1,b,i,j,f}}\;
\mathbf{W}_{1,u,v,c,f}.
$$

**Outgoing shapes.** $\nabla_{\mathbf{W}_1}$: $(3,3,1,8)$ = 72 values;
$\nabla_{\boldsymbol{\beta}_1}$: $(8,)$; $\nabla_{\mathbf{X}}$: $(32,28,28,1)$.

**Parameter gradients.** 80 in total.
**Propagated.** $\nabla_{\mathbf{X}}$ leaves the network — it is the input
gradient, and the first layer has no earlier layer to receive it.

**Accumulation is the whole difficulty here.** Each pixel belongs to up to nine
receptive fields:

- an interior pixel such as $X[5,5]$ is covered by nine output positions,
  $(i,j) \in \{3,4,5\}^2$;
- a pixel one step from the border, such as $X[1,5]$, by six ($i \in \{0,1\}$,
  $j \in \{3,4,5\}$);
- a pixel on the border itself, such as $X[0,5]$, by three ($i = 0$ only);
- a corner such as $X[0,0]$ by exactly one.

Therefore $\nabla_{\mathbf{W}_1}$ sums over all 32 × 26 × 26 positions (the same
filter is reused at every position), and $\nabla_{\mathbf{X}}$ must **accumulate**
over all positions covering a pixel. Writing `dX[..., h, w, c] = contribution`
instead of `+=` keeps only the last window and produces a gradient that passes a
loose check on corners but fails everywhere else. [Section 19.2](cnn_mathematical_foundations.md#192-convolution-backward-dw-accumulation)
shows the accumulation numerically.

---

### Stage 7 — The input gradient

$\nabla_{\mathbf{X}}$ has shape $(32, 28, 28, 1)$, identical to the input. It is
**not** used by SGD (the input is not a parameter), but it is the value that
proves the convolution backward pass is correct: comparing it with finite
differences of the loss with respect to $\mathbf{X}$ tests the whole chain from
the loss down to the first layer. If $\nabla_{\mathbf{X}}$ is right, every
intermediate gradient is right as well, because each stage had to be correct to
produce it.

---

## 4. Compact shape trace

**Single image ($B = 1$)**

| Stage | Forward output | Backward gradient arriving | Backward gradient produced | Parameters |
|---|---|---|---|---|
| input | $(28,28,1)$ | — | $(28,28,1)$ | 0 |
| conv | $(26,26,8)$ | $(26,26,8)$ | $\nabla\mathbf{W}_1 (3,3,1,8)$, $\nabla\mathbf{b}_1 (8,)$ | 80 |
| ReLU | $(26,26,8)$ | $(26,26,8)$ | $(26,26,8)$ | 0 |
| maxpool | $(13,13,8)$ | $(13,13,8)$ | $(26,26,8)$ | 0 |
| flatten | $(1352,)$ | $(1352,)$ | $(13,13,8)$ | 0 |
| dense | $(10,)$ | $(10,)$ | $\nabla\mathbf{W}_2 (1352,10)$, $\nabla\mathbf{b}_2 (10,)$, $(1352,)$ | 13,530 |
| softmax | $(10,)$ | $(10,)$ | $(10,)$ (combined with the loss) | 0 |
| loss | scalar | 1 | — | 0 |

**Batch ($B = 32$)** — the same table with every shape prefixed by 32, except the
parameter gradients, which do **not** grow with the batch:

| Stage | Forward output | Backward gradient arriving | Backward gradient produced |
|---|---|---|---|
| input | $(32,28,28,1)$ | — | $(32,28,28,1)$ |
| conv | $(32,26,26,8)$ | $(32,26,26,8)$ | $\nabla\mathbf{W}_1 (3,3,1,8)$, $\nabla\mathbf{b}_1 (8,)$ |
| ReLU | $(32,26,26,8)$ | $(32,26,26,8)$ | $(32,26,26,8)$ |
| maxpool | $(32,13,13,8)$ | $(32,13,13,8)$ | $(32,26,26,8)$ |
| flatten | $(32,1352)$ | $(32,1352)$ | $(32,13,13,8)$ |
| dense | $(32,10)$ | $(32,10)$ | $\nabla\mathbf{W}_2 (1352,10)$, $\nabla\mathbf{b}_2 (10,)$, $(32,1352)$ |
| softmax + loss | $(32,10)$ → scalar | $(32,10)$ | $(32,10)$ |

**Gradient shapes versus parameter shapes.** Every parameter gradient has exactly
the shape of its parameter; every propagated gradient has exactly the shape of
the forward output it corresponds to. A mismatch on either side is a bug, not a
tolerable difference.

---

## 5. One step, in order (mathematical pseudocode)

Not an implementation — the order and the shapes are the point.

```text
for each mini-batch (X, y):                       shapes

    # ---- forward, caching what backward needs ----
    Z1        = conv2d(X, W1, b1)                 (32,26,26,8)
    A1        = relu(Z1)                          cache Z1 (or mask)
    A2, argmx = maxpool2d(A1, 2, 2)               cache argmx (32,13,13,8)
    A3        = flatten(A2)                       cache shape (32,13,13,8)
    Z3        = A3 @ W2 + b2                      cache A3, W2
    P         = softmax_stable(Z3)                (32,10)
    L         = -mean(log P[b, y_b])              scalar

    # ---- backward, right to left ----
    G   = (P - onehot(y)) / B                     (32,10)
    dW2 = A3.T @ G                                (1352,10)
    db2 = sum(G, axis=0)                          (10,)
    dA3 = G @ W2.T                                (32,1352)
    dA2 = reshape(dA3, (32,13,13,8))              (32,13,13,8)
    dA1 = maxpool_backward(dA2, argmx,
                           shape=(32,26,26,8))    (32,26,26,8)
    dZ1 = dA1 * (Z1 > 0)                          (32,26,26,8)
    dW1, db1, dX = conv2d_backward(dZ1, X, W1)    (3,3,1,8), (8,), (32,28,28,1)

    # ---- one SGD update, after every gradient exists ----
    W1 -= lr * dW1 ;  b1 -= lr * db1
    W2 -= lr * dW2 ;  b2 -= lr * db2
```

Chapter order matters in two places: the ReLU mask must come from `Z1`, not
`A1`, and all four parameter gradients must be computed before the first update.

---

## 6. Maintaining a correct implementation in Phase 3

Backpropagation is easy to get *approximately* right and hard to get exactly
right. The checks below are the plan for Phase 3; none of them runs yet.

1. **Analytic versus numerical gradients.** For each parameter tensor, perturb a
   handful of entries by $\pm\varepsilon$ and compare

   $$
   \frac{L(\theta+\varepsilon) - L(\theta-\varepsilon)}{2\varepsilon}
   \quad\text{with}\quad
   \nabla_\theta L,
   $$

   requiring a relative error below about $10^{-6}$. Do this in **float64** on a
   small batch: float32 differences are dominated by rounding once
   $\varepsilon$ is small ($\varepsilon \approx 10^{-4}$ is the practical choice
   for float64 here). Check the convolution weights, both biases,
   $\nabla_{\mathbf{A}_3}$, $\nabla_{\mathbf{Z}_1}$ and $\nabla_{\mathbf{X}}$.
2. **Structural invariants.** Every gradient has the shape of its variable; each
   row of $\mathbf{G}$ sums to 0; pooling preserves the total,
   $\sum \nabla_{\mathbf{A}_1} = \sum \nabla_{\mathbf{A}_2}$; and the convolution's
   bias gradient must equal the sum of $\nabla_{\mathbf{Z}_1}$ over the batch and
   both spatial axes. These assertions cost nothing and catch most routing bugs.
3. **Decreasing loss on a single batch.** Overfit one fixed batch of 32 for a few
   dozen steps: the loss must fall monotonically. This is a correctness check,
   not a result — no accuracy figure is reported from it.
4. **Determinism.** The same seed must reproduce the same initial parameters and
   the same loss sequence, extending the Phase 1 guarantee to training.
5. **Agreement with Keras.** Phase 5 initialises Keras with the NumPy-drawn
   weights and compares logits per sample; matching to float tolerance is the
   strongest end-to-end evidence that both implementations compute the same
   function.

---

## 7. Common mistakes this walkthrough is designed to prevent

| Mistake | Symptom | Fix |
|---|---|---|
| Dividing by $B$ twice (loss and update) | loss barely moves, effective lr 32× too small | apply $1/B$ once, in $\mathbf{G}$ |
| ReLU mask from $\mathbf{A}_1$ instead of $\mathbf{Z}_1$ | gradient leaks where $z<0$ and $z=0$ | store $\mathbf{Z}_1$ or `Z1 > 0` |
| Recomputing the pool argmax in backward | wrong routing if the layer is reused | cache argmax during forward |
| `dX[...] =` instead of `+=` in convolution | gradient correct only at corners | accumulate over all covering windows |
| `dW[...] =` instead of `+=` in convolution | only the last spatial position counted | sum over batch and positions |
| Channel-first reshape after flatten | trains, but gradient check fails | reshape to the cached input shape |
| Softmax without the row-max shift | `nan` from `inf/inf` | subtract the per-row maximum |
| Transposed dense weights | shape error, or silently wrong function | store $\mathbf{W}_2$ as $(M,N)$, use $\mathbf{A}^{\top}\mathbf{G}$ |
| Updating weights between parameter gradients | not SGD; gradients from different functions | compute all four gradients, then update |
| Integer labels compared to probabilities | wrong loss, often 0 accuracy | one-hot encode to $\mathbf{Y}$, or index $\mathbf{P}[b, y_b]$ |
| Forgetting the convolution bias in the count | 72 instead of 80 | bias per filter, counted in `architecture.py` |

---

## 8. Consistency with the rest of the repository

| Fact | Where it is defined | Where it is checked |
|---|---|---|
| Layer sequence and configurations | `configs/experiment_config.json` | `tests/test_architecture.py` |
| Output shapes $(26,26,8) \to (13,13,8) \to 1352 \to 10$ | `models/architecture.py` | `tests/test_architecture.py` |
| 80 + 13,530 = 13,610 parameters | `models/architecture.py` | `tests/test_architecture.py` |
| Notation and derivations | `docs/cnn_mathematical_foundations.md` | reviewed against this document |
| Seed 42, SGD, lr 0.01, batch 32, 5 epochs, mean CE | `configs/experiment_config.json` | `tests/test_architecture.py` |

Nothing in this document describes code that exists: there is no forward pass,
no backward pass, no optimiser and no training loop in the repository yet.
