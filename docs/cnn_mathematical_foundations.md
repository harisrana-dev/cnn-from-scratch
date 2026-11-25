# CNN Mathematical Foundations

This document derives every operation the scratch NumPy CNN performs. It is the
reference for Phase 3 (layer implementations), Phase 4 (training) and the viva:
each equation below is written so it can be translated into code line by line.

**Scope.** Mathematics, shapes and parameter counts only. No layer has been
implemented yet, no training has run, and this document quotes **no accuracy,
loss or timing result**. Results will be added in Phase 6 after they are
measured.

Companion document: [`backpropagation_walkthrough.md`](backpropagation_walkthrough.md)
traces one full training step backwards through the frozen architecture.

---

## Table of contents

1. [Notation](#1-notation)
2. [CNN overview and the forward pipeline](#2-cnn-overview-and-the-forward-pipeline)
3. [Convolution forward propagation](#3-convolution-forward-propagation)
4. [Cross-correlation versus mathematical convolution](#4-cross-correlation-versus-mathematical-convolution)
5. [Convolution output dimensions](#5-convolution-output-dimensions)
6. [ReLU forward and backward](#6-relu-forward-and-backward)
7. [Max pooling forward](#7-max-pooling-forward)
8. [Max pooling backward](#8-max-pooling-backward)
9. [Flatten forward and backward](#9-flatten-forward-and-backward)
10. [Dense layer forward](#10-dense-layer-forward)
11. [Dense layer gradients](#11-dense-layer-gradients)
12. [Softmax and numerical stability](#12-softmax-and-numerical-stability)
13. [Categorical cross-entropy](#13-categorical-cross-entropy)
14. [Combined softmax-cross-entropy derivative](#14-combined-softmax-cross-entropy-derivative)
15. [Mini-batch gradient averaging](#15-mini-batch-gradient-averaging)
16. [SGD parameter updates](#16-sgd-parameter-updates)
17. [Parameter counting](#17-parameter-counting)
18. [He initialisation](#18-he-initialisation)
19. [Worked numerical examples](#19-worked-numerical-examples)
20. [References](#20-references)

---

## 1. Notation

The same symbols are used throughout this document and in
`backpropagation_walkthrough.md`.

**Index conventions**

| Symbol | Meaning |
|---|---|
| $B$ | batch size (32 in the initial configuration) |
| $H, W, C$ | input height, width, channels (28, 28, 1) |
| $K$ | convolution kernel size (3) |
| $S$ | stride (1 for convolution, 2 for pooling) |
| $P$ | number of implicit zero rows/columns added per side (0, `valid` padding) |
| $F$ | number of convolution filters (8) |
| $M$ | number of dense inputs after flattening (1352) |
| $N$ | number of output classes / dense outputs (10) |
| $i, j$ | output row and column index |
| $u, v$ | kernel row and column offset, $0 \le u,v < K$ |
| $c$ | input channel index, $0 \le c < C$ |
| $f$ | filter / output channel index, $0 \le f < F$ |
| $b$ | index of a sample in the mini-batch, $0 \le b < B$ |
| $k$ | class index, $0 \le k < N$ |
| $m$ | index along the flattened feature vector |

**Types of quantity**

| Symbol | Type | Meaning |
|---|---|---|
| $\mathbf{X}$ | 4-D tensor $(B, H, W, C)$ | mini-batch of input images |
| $\mathbf{W}^{c}, \mathbf{W}$ | 4-D tensor $(K,K,C,F)$, 2-D matrix $(M,N)$ | trainable weights |
| $\boldsymbol{\beta}$ | vector $(F,)$ or $(N,)$ | trainable biases |
| $\mathbf{Z}$ | tensor | pre-activation (linear) output |
| $\mathbf{A}$ | tensor | activation (post-ReLU / flattened) |
| $\mathbf{Y}$ | tensor $(B, N)$ | one-hot encoded targets |
| $\mathbf{P}$ | tensor $(B, N)$ | predicted class probabilities |
| $L$ | scalar | mini-batch loss |
| $\nabla_\theta L$ | same shape as $\theta$ | gradient of the loss w.r.t. parameter $\theta$ |
| $\odot$ | — | element-wise (Hadamard) product |
| $\mathbb{1}[\cdot]$ | — | indicator, 1 when the condition holds, else 0 |

Superscripts in round brackets, e.g. $\mathbf{Z}^{(2)}$, refer to a layer index
when a symbol is reused. The frozen architecture has only two parameterised
layers, so plain $\mathbf{W}$/$\boldsymbol{\beta}$ with a subscript is used
instead: $(\mathbf{W}_1, \boldsymbol{\beta}_1)$ for the convolution,
$(\mathbf{W}_2, \boldsymbol{\beta}_2)$ for the dense layer.

**Overloaded symbols, spelled out.** $\beta$ is always a bias, never a
probability; $b$ is always a batch index. $B$ is the batch size and $\mathbb{1}$
is the indicator. $P$ is padding and $\mathbf{P}$ is the probability matrix —
the boldface distinguishes them.

---

## 2. CNN overview and the forward pipeline

A convolutional network composes linear maps that see small spatial
neighbourhoods (convolution) with fixed non-linearities (ReLU) and fixed
down-sampling (max pooling), then hands a compact feature vector to a dense
classifier. Only convolution and dense weights are learned; everything else is
fixed arithmetic.

The architecture is frozen in `configs/experiment_config.json` and validated by
`src/cnn_scratch/models/architecture.py`, so the NumPy and Keras models cannot
silently diverge.

| # | Layer | Configuration | Output (one sample) | Output (batch, $B=32$) | Trainable parameters |
|---|---|---|---|---|---|
| 0 | Input | Fashion-MNIST grayscale, NHWC | $(28, 28, 1)$ | $(32, 28, 28, 1)$ | 0 |
| 1 | Conv2D | 8 filters, $3\times3$, stride 1, `valid`, bias | $(26, 26, 8)$ | $(32, 26, 26, 8)$ | 80 |
| 2 | ReLU | element-wise $\max(0, z)$ | $(26, 26, 8)$ | $(32, 26, 26, 8)$ | 0 |
| 3 | MaxPool2D | $2\times2$, stride 2, `valid` | $(13, 13, 8)$ | $(32, 13, 13, 8)$ | 0 |
| 4 | Flatten | row-major reshape | $(1352,)$ | $(32, 1352)$ | 0 |
| 5 | Dense | 10 units, bias, raw logits | $(10,)$ | $(32, 10)$ | 13,530 |
| 6 | Softmax | 10 class probabilities | $(10,)$ | $(32, 10)$ | 0 |

**Total trainable parameters: 13,610.** A plain count of float32 values is
$13{,}610 \times 4 = 54{,}440$ bytes of parameters.

The forward pass is the composition

$$
\mathbf{P} \;=\;
\operatorname{softmax}\!\Big(
\operatorname{flatten}\big(
\operatorname{maxpool}\big(
\operatorname{ReLU}\big(
\operatorname{conv}(\mathbf{X}; \mathbf{W}_1, \boldsymbol{\beta}_1)
\big)\big)\big)\,
\mathbf{W}_2 + \boldsymbol{\beta}_2
\Big),
$$

and the loss compares $\mathbf{P}$ with the one-hot targets $\mathbf{Y}$:

$$
\mathbf{X} \xrightarrow{\text{conv}} \mathbf{Z}^{(1)} \xrightarrow{\text{ReLU}}
\mathbf{A}^{(1)} \xrightarrow{\text{pool}} \mathbf{A}^{(2)} \xrightarrow{\text{flatten}}
\mathbf{A}^{(3)} \xrightarrow{\text{dense}} \mathbf{Z}^{(3)} \xrightarrow{\text{softmax}}
\mathbf{P} \xrightarrow{\text{CE}} L .
$$

`valid` padding means no zeros are added, so the spatial size shrinks at every
windowed layer: $28 \to 26 \to 13$. The batch dimension is never mixed: each
sample is transformed independently, which is what makes mini-batch training a
plain stack of independent samples.

Layout is NHWC (batch, height, width, channels) everywhere, matching the arrays
produced in Phase 1 (`(n, 28, 28, 1)`). This is also Keras' default
`channels_last` layout, so no transposition is needed to compare the models.

---

## 3. Convolution forward propagation

For a single filter $f$, the convolution slides a $K \times K$ window over every
channel of the input, multiplies element-wise and sums. With padding $P$ and
stride $S$:

$$
\mathbf{Z}_{b,i,j,f} \;=\; \beta_f \;+\;
\sum_{c=0}^{C-1} \sum_{u=0}^{K-1} \sum_{v=0}^{K-1}
\mathbf{X}_{b,\;iS+u-P,\;jS+v-P,\;c}\;\; \mathbf{W}^{c}_{u,v,c,f},
$$

where any $\mathbf{X}$ index lying outside $[0, H)$ or $[0, W)$ contributes 0
(only relevant when $P > 0$; with `valid` padding no index ever falls outside).

For the frozen layer: $C = 1$, $K = 3$, $S = 1$, $P = 0$, $F = 8$, so

$$
\mathbf{Z}_{b,i,j,f} \;=\; \beta_f \;+\;
\sum_{u=0}^{2}\sum_{v=0}^{2}
\mathbf{X}_{b,\,i+u,\,j+v,\,0}\; \mathbf{W}_{u,v,0,f},
\qquad i,j \in [0, 26).
$$

**Receptive field.** Each output value depends on a $3\times3\times1 = 9$-element
patch of the input and on the 9 weights of one filter. The same 9 weights are
reused at all $26 \times 26$ positions (weight sharing), and the 8 filters each
learn a different pattern over the same patch. That sharing is why the layer has
only 80 parameters instead of the $26\cdot26\cdot9 = 6{,}084$ a fully connected
layer of the same width would need.

**Why "valid".** With $P = 0$ every output is computed from real pixels only, so
no artificial border has to be interpreted, and the arithmetic is
$26 = 28 - 3 + 1$ exactly. It is also the simplest case to hand-verify in a viva.

---

## 4. Cross-correlation versus mathematical convolution

In signal processing, the 2-D convolution of an image $X$ with a kernel $W$ is

$$
(X * W)_{i,j} \;=\; \sum_{u}\sum_{v} X_{i-u,\;j-v}\, W_{u,v},
$$

which is the same as a cross-correlation with the kernel flipped in both spatial
dimensions. What the equation in [Section 3](#3-convolution-forward-propagation)
computes — and what NumPy code, `keras.layers.Conv2D`, PyTorch `Conv2d` and
every mainstream deep-learning framework compute — is cross-correlation:

$$
(X \star W)_{i,j} \;=\; \sum_{u}\sum_{v} X_{i+u,\;j+v}\, W_{u,v}.
$$

There is no flip. The three reasons this is the convention:

1. **The filters are learned, so the flip is absorbed.** Training a
   cross-correlation with kernel $W$ is exactly equivalent to training a true
   convolution with kernel $W'$ where $W'_{u,v} = W_{K-1-u,\,K-1-v}$. The
   hypothesis class is identical; only the stored array differs. Since the
   network learns its own kernels, nothing is lost.
2. **The gradients are simple.** Cross-correlation's backward pass reuses the
   forward indexing directly (Section 3 of the walkthrough). A flip would have
   to be applied consistently in both directions, which is an easy place to
   introduce a silent bug.
3. **It matches the library behaviour we compare against.** Keras does not flip,
   so if the scratch implementation flipped, the two models would learn mirrored
   kernels and their weights could not be compared even though their outputs
   might still match.

**Practical consequence for this project.** The derivative formulas below are
for cross-correlation. If a result is ever cross-checked against
`scipy.signal.convolve2d`, that function *does* flip, so the kernel must be
flipped or `scipy.signal.correlate2d` used instead, otherwise the comparison is
wrong. For a symmetric kernel (e.g. all-ones averaging) the two operations
coincide, which is why the difference is easy to miss in toy examples.

**Terminology.** This document follows the deep-learning convention and calls
the cross-correlation operation "convolution".

---

## 5. Convolution output dimensions

For an input axis of length $n$, a window of length $k$, stride $s$ and padding
$p$ per side, the number of valid window positions is

$$
n_{\text{out}} \;=\; \left\lfloor \frac{n + 2p - k}{s} \right\rfloor + 1 .
$$

Applied to the frozen architecture:

| Layer | Input axis | $k$ | $s$ | $p$ | Output axis |
|---|---|---|---|---|---|
| Conv2D | 28 | 3 | 1 | 0 | $\lfloor (28-3)/1 \rfloor + 1 = 26$ |
| MaxPool2D | 26 | 2 | 2 | 0 | $\lfloor (26-2)/2 \rfloor + 1 = 13$ |

The formula is implemented once, in
`src/cnn_scratch/models/architecture.py::conv_output_size`, and is asserted by
`tests/test_architecture.py` for several $(n, k, s, p)$ combinations.

Flattening multiplies the remaining axes:

$$
M \;=\; 13 \times 13 \times 8 \;=\; 1352 .
$$

The full-width layer would need $1352 \times 10 + 10 = 13{,}530$ parameters,
which is why the spatial size is reduced before flattening: pooling at stride 2
quarters the number of activations and therefore quarters the dense weight
count.

---

## 6. ReLU forward and backward

**Forward.** Applied element-wise, independently per value:

$$
\mathbf{A}^{(1)}_{b,i,j,f} \;=\; \max\!\left(0,\; \mathbf{Z}^{(1)}_{b,i,j,f}\right).
$$

ReLU is non-linear, cheap, and its derivative is exactly 0 or 1, which keeps
gradients from shrinking the way sigmoid derivatives do (the vanishing-gradient
problem). It also makes the layer sparse: roughly half the activations are
exactly zero.

**Backward.** With $\partial L/\partial \mathbf{A}^{(1)}$ given by the layer that
follows,

$$
\frac{\partial L}{\partial \mathbf{Z}^{(1)}_{b,i,j,f}}
\;=\;
\frac{\partial L}{\partial \mathbf{A}^{(1)}_{b,i,j,f}}
\cdot
\mathbb{1}\!\left[\mathbf{Z}^{(1)}_{b,i,j,f} > 0\right].
$$

So an upstream gradient passes through unchanged where the pre-activation was
positive and is blocked where it was negative.

**The point $z = 0$.** ReLU is not differentiable at 0. The project uses the
subgradient $\mathbb{1}[z > 0]$, i.e. the derivative at exactly 0 is **0**. This
is the NumPy convention implemented by `(z > 0)` and the one Keras uses, so both
models behave identically. The choice matters only on exact zeros, which are
measure-zero in practice but do occur after ReLU (for the *next* layer's input).

**What must be cached.** Only the pre-activation $\mathbf{Z}^{(1)}$ (or
equivalently the boolean mask $\mathbf{Z}^{(1)} > 0$), never the post-activation
$\mathbf{A}^{(1)}$. Using the post-activation would give the wrong mask wherever
$z < 0$ produced $a = 0$ and elsewhere the two agree only by coincidence.

ReLU has no trainable parameters.

---

## 7. Max pooling forward

Max pooling has no weights: it partitions each channel into non-overlapping
$2\times2$ windows (stride 2, `valid`) and keeps the largest value.

$$
\mathbf{A}^{(2)}_{b,i,j,f} \;=\;
\max_{\substack{0 \le u < 2 \\ 0 \le v < 2}}
\mathbf{X}^{(1)\vphantom{(2)}}_{b,\;2i+u,\;2j+v,\;f},
$$

where $\mathbf{X}^{(1)}$ denotes the ReLU output (input to the pooling layer).
Channels are pooled independently, so the channel count is preserved:
$8 \to 8$.

For each of the $13 \times 13 \times 8$ output values the layer must remember
**which** of the four inputs won:

$$
(\hat{u}, \hat{v})_{b,i,j,f} \;=\;
\argmax_{\substack{0 \le u < 2 \\ 0 \le v < 2}}
\mathbf{X}^{(1)}_{b,\;2i+u,\;2j+v,\;f}.
$$

This is the **argmax cache**; it is the entire backward-pass state of the layer
(a boolean mask of the same shape as the input works equally well).

**Deterministic tie-breaking.** When several inputs in a window are equal,
the project selects the **first maximum in row-major order**, i.e. the smallest
$(u, v)$ in lexicographic order — top-left wins over top-right, bottom-left,
bottom-right. This is exactly what `numpy.argmax` returns for a flattened
window, so the rule costs nothing to implement and is reproducible on every run.
A random tie-break would make two runs with the same seed diverge, which would
break the Phase 1 determinism guarantee.

**Why pooling.** It halves each spatial axis, cutting the feature map from 5408
to 1352 values per sample, and makes the representation locally translation
invariant: a small shift inside a window usually leaves the maximum unchanged.
Windows do not overlap at stride 2, so each input value feeds exactly one output.

---

## 8. Max pooling backward

Max pooling only routes gradients; it never scales them. Each output gradient is
sent to the single input position that was the maximum, and every other position
in the window receives exactly zero:

$$
\frac{\partial L}{\partial \mathbf{X}^{(1)}_{b,\,h,\,w,\,f}}
\;=\;
\sum_{i,j \;:\; (2i+\hat{u},\;2j+\hat{v}) = (h,w)}
\frac{\partial L}{\partial \mathbf{A}^{(2)}_{b,i,j,f}} .
$$

In practice the implementation scatters with the cached positions:

$$
\frac{\partial L}{\partial \mathbf{X}^{(1)}}_{b,\;2i+\hat{u}_{b,i,j,f},\;2j+\hat{v}_{b,i,j,f},\;f}
\;\mathrel{+}=\;
\frac{\partial L}{\partial \mathbf{A}^{(2)}_{b,i,j,f}},
$$

and leaves the other $3/4$ of the input gradient at zero. Implemented as a mask
multiply, it is

$$
\frac{\partial L}{\partial \mathbf{X}^{(1)}} \;=\;
\operatorname{unpool}\!\left(\frac{\partial L}{\partial \mathbf{A}^{(2)}},\;
\hat{u}, \hat{v}\right).
$$

**Consequences to state in the viva.**

- The gradient is **not** divided by the window size and **not** averaged. Only
  the selected position receives gradient, so the total sum of the input
  gradient equals the total sum of the output gradient.
- The argmax must come from the **forward** pass, not recomputed from the
  current input: the parameters have not changed within a step, but recomputing
  would be wasteful and would break the moment the layer is reused after a
  weight update.
- Because windows do not overlap at stride 2, each input receives at most one
  gradient contribution. With overlapping windows (stride < pool size) a single
  input can be the maximum of several windows and the contributions must be
  **accumulated** ($\mathrel{+}=$), not assigned — the same accumulation issue as
  in the convolution backward pass.

Pooling has no trainable parameters.

---

## 9. Flatten forward and backward

**Forward.** Row-major (C-order) reshape, no data copied conceptually:

$$
\mathbf{A}^{(3)}_{b,\,m} \;=\; \mathbf{A}^{(2)}_{b,\,i,\,j,\,f},
\qquad
m \;=\; (i \cdot 13 + j) \cdot 8 + f .
$$

With $i \in [0,13)$, $j \in [0,13)$, $f \in [0,8)$ the index $m$ ranges over
$[0, 1352)$ — every element appears exactly once, so flattening is a bijection
between the tensor and the vector.

**Backward.** The exact inverse reshape:

$$
\frac{\partial L}{\partial \mathbf{A}^{(2)}} \;=\;
\operatorname{reshape}\!\left(
\frac{\partial L}{\partial \mathbf{A}^{(3)}},\;
(B, 13, 13, 8)\right).
$$

Because the mapping is a pure re-indexing, the gradient of each element is
identical to the gradient of the element it became; nothing is summed or scaled.

**The classic bug.** The dense layer and the convolution must agree on the same
ordering. If the backward reshape uses a different convention (for example
$(B, 8, 13, 13)$, i.e. channels-first) the network still trains — the dense
weights simply learn a permuted input — but the gradient check against finite
differences fails and the two implementations no longer match. Storing the
input shape at forward time and reshaping back to it is the robust fix.

Flatten has no trainable parameters.

---

## 10. Dense layer forward

With $\mathbf{A}^{(3)} \in \mathbb{R}^{B \times M}$, $\mathbf{W}_2 \in
\mathbb{R}^{M \times N}$ and $\boldsymbol{\beta}_2 \in \mathbb{R}^{N}$:

$$
\mathbf{Z}^{(3)}_{b,k} \;=\; \beta_{2,k} \;+\; \sum_{m=0}^{M-1}
\mathbf{A}^{(3)}_{b,m} \, \mathbf{W}_{2,m,k},
\qquad\text{i.e.}\qquad
\mathbf{Z}^{(3)} = \mathbf{A}^{(3)} \mathbf{W}_2 + \mathbf{1}_B \boldsymbol{\beta}_2^{\!\top}.
$$

The weight matrix is stored as $(M, N) = (1352, 10)$ — **input by output** — so
the forward pass needs no transpose and each column of $\mathbf{W}_2$ holds the
weights of one class. The bias is broadcast across the batch ($\mathbf{1}_B$ is a
column of $B$ ones); it is not a parameter per sample.

The layer is linear: $\mathbf{Z}^{(3)}$ is the raw score (logit) for each class,
and the softmax in [Section 12](#12-softmax-and-numerical-stability) turns those
scores into probabilities. Keras calls this `Dense(10)` with the default linear
activation, which is the same thing.

Cost: $B \times 1352 \times 10$ multiply-adds per forward pass, i.e.
432,640 for $B = 32$.

---

## 11. Dense layer gradients

Given $\mathbf{G} \equiv \partial L/\partial \mathbf{Z}^{(3)} \in
\mathbb{R}^{B \times N}$ from the softmax/CE stage:

**Weights.** Each $\mathbf{W}_{2,m,k}$ is used once per sample, so it accumulates
one contribution per sample in the batch:

$$
\frac{\partial L}{\partial \mathbf{W}_{2,m,k}}
\;=\; \sum_{b=0}^{B-1} \mathbf{A}^{(3)}_{b,m}\, \mathbf{G}_{b,k}
\qquad\Longleftrightarrow\qquad
\nabla_{\mathbf{W}_2} L \;=\; \mathbf{A}^{(3)\top} \mathbf{G},
$$

with shape $(M, N) = (1352, 10)$ — identical to $\mathbf{W}_2$, as required.

**Biases.** $\beta_{2,k}$ is added to every sample, so its gradient is the sum
over the batch:

$$
\frac{\partial L}{\partial \beta_{2,k}} \;=\; \sum_{b=0}^{B-1} \mathbf{G}_{b,k}
\qquad\Longleftrightarrow\qquad
\nabla_{\boldsymbol{\beta}_2} L \;=\; \mathbf{G}^{\top} \mathbf{1}_B ,
$$

with shape $(N,) = (10,)$.

**Inputs.** Propagated to the flatten layer:

$$
\frac{\partial L}{\partial \mathbf{A}^{(3)}_{b,m}}
\;=\; \sum_{k=0}^{N-1} \mathbf{G}_{b,k}\,\mathbf{W}_{2,m,k}
\qquad\Longleftrightarrow\qquad
\nabla_{\mathbf{A}^{(3)}} L \;=\; \mathbf{G}\,\mathbf{W}_2^{\top},
$$

with shape $(B, M) = (32, 1352)$.

**Shape rule of thumb.** For $\mathbf{Z} = \mathbf{A}\mathbf{W} + \boldsymbol{\beta}$:
$\nabla_{\mathbf{W}} = \mathbf{A}^{\top}\mathbf{G}$ (both inputs' indexes kept),
$\nabla_{\mathbf{A}} = \mathbf{G}\mathbf{W}^{\top}$ (both outputs' indexes kept),
$\nabla_{\boldsymbol{\beta}}$ = sum over samples. Every gradient has the shape of
the parameter it belongs to; if it does not, the implementation is wrong.

---

## 12. Softmax and numerical stability

Softmax maps $N$ unconstrained logits to a probability distribution:

$$
\mathbf{P}_{b,k} \;=\;
\frac{\exp\!\left(\mathbf{Z}^{(3)}_{b,k}\right)}
{\sum_{l=0}^{N-1}\exp\!\left(\mathbf{Z}^{(3)}_{b,l}\right)},
\qquad
\sum_{k}\mathbf{P}_{b,k} = 1,\;\; \mathbf{P}_{b,k} > 0 .
$$

It depends only on **differences** between logits: adding a constant $m$ to every
logit cancels in numerator and denominator. That is exactly the stability trick:

$$
\mathbf{P}_{b,k} \;=\;
\frac{\exp\!\left(\mathbf{Z}^{(3)}_{b,k} - m_b\right)}
{\sum_{l}\exp\!\left(\mathbf{Z}^{(3)}_{b,l} - m_b\right)},
\qquad m_b = \max_k \mathbf{Z}^{(3)}_{b,k}.
$$

**Why the shift is mandatory in float32.** $\exp(89) \approx 4.5\times10^{38}$
already exceeds the float32 maximum ($\approx 3.4\times10^{38}$), and
$\exp(-104)$ underflows to 0. Unshifted, a logit of 100 would produce `inf/inf`
(nan) and a logit of $-100$ in every class would produce `0/0`. After
subtracting the row maximum the largest argument is exactly 0, so the largest
numerator is 1 and the denominator lies in $[1, N]$: no overflow, and the
smallest representable probability is limited only by underflow of the
individual numerators, which contributes $\exp(-88) \approx 0$ harmlessly.

The maximum is taken **per sample** (along the class axis), not over the whole
batch, because each row is an independent distribution.

**Log form.** The loss uses $\log \mathbf{P}_{b,k} = \mathbf{Z}^{(3)}_{b,k} -
m_b - \log \sum_l \exp(\mathbf{Z}^{(3)}_{b,l} - m_b)$ (log-sum-exp), which never
takes the log of a rounded probability. With a one-hot target only one term is
needed:

$$
\log \mathbf{P}_{b,y_b} \;=\; \mathbf{Z}^{(3)}_{b,y_b} - m_b - \log\sum_l \exp\!\left(\mathbf{Z}^{(3)}_{b,l} - m_b\right).
$$

**Implementation note.** Guard the logarithm anyway (`np.clip` at the smallest
normal float32, or use the log-sum-exp form) so that a degenerate probability of
exactly 0 cannot turn the loss into `inf`.

---

## 13. Categorical cross-entropy

For one sample with one-hot target $\mathbf{Y}_{b,:}$ and prediction
$\mathbf{P}_{b,:}$:

$$
L_b \;=\; -\sum_{k=0}^{N-1} \mathbf{Y}_{b,k} \log \mathbf{P}_{b,k}
\;=\; -\log \mathbf{P}_{b,\,y_b},
$$

because only the true class term survives the one-hot sum. The mini-batch loss
is the **mean** over samples:

$$
L \;=\; \frac{1}{B}\sum_{b=0}^{B-1} L_b
\;=\; -\frac{1}{B}\sum_{b=0}^{B-1} \log \mathbf{P}_{b,\,y_b}.
$$

**Properties.** $L_b \ge 0$, and $L_b = 0$ only for a confident correct
prediction ($\mathbf{P}_{b,y_b} = 1$). Cross-entropy punishes confident mistakes
without bound ($\mathbf{P}_{b,y_b} \to 0 \Rightarrow L_b \to \infty$), which is
the behaviour we want from a classifier loss; squared error saturates instead and
combines badly with softmax.

**Why the mean and not the sum.** With a mean, the gradient magnitude does not
depend on the batch size, so the learning rate keeps its meaning if the batch
size changes later. Keras' `categorical_crossentropy` also averages over the
batch by default, so the two implementations use the same objective — this is
recorded as `"loss_reduction": "mean"` in the configuration.

**Labels.** Phase 1 stores integer labels (`uint8`). One-hot encoding to
$\mathbf{Y} \in \{0,1\}^{B \times N}$ happens only at training time, as the
Phase 1 specification requires. Equivalently, the loss can be written as
$-\frac{1}{B}\sum_b \log \mathbf{P}_{b,y_b}$ using the integer index directly,
which is what the scratch implementation will do; the two forms are identical.

---

## 14. Combined softmax-cross-entropy derivative

The softmax Jacobian is dense: $\partial \mathbf{P}_{b,k}/\partial
\mathbf{Z}^{(3)}_{b,l} = \mathbf{P}_{b,k}(\delta_{kl} - \mathbf{P}_{b,l})$.
Chaining that with the cross-entropy derivative simplifies dramatically. For a
single sample:

$$
\frac{\partial L_b}{\partial \mathbf{Z}^{(3)}_{b,k}}
\;=\; \mathbf{P}_{b,k} - \mathbf{Y}_{b,k}.
$$

**Derivation.** Write $L_b = -\log \mathbf{P}_{b,y_b}$ with
$\log \mathbf{P}_{b,k} = \mathbf{Z}_{b,k} - \log S_b$ and
$S_b = \sum_l \exp \mathbf{Z}_{b,l}$. Then
$\partial \log \mathbf{P}_{b,k}/\partial \mathbf{Z}_{b,l} = \delta_{kl} - \mathbf{P}_{b,l}$,
so

$$
\frac{\partial L_b}{\partial \mathbf{Z}_{b,l}}
= -\left(\delta_{l,y_b} - \mathbf{P}_{b,l}\right)
= \mathbf{P}_{b,l} - \mathbf{Y}_{b,l}. \qquad\blacksquare
$$

**For the mini-batch with mean loss** the $1/B$ factor survives:

$$
\boxed{\;\mathbf{G} \;=\; \frac{\partial L}{\partial \mathbf{Z}^{(3)}}
\;=\; \frac{1}{B}\left(\mathbf{P} - \mathbf{Y}\right)\;}
\qquad \mathbf{G} \in \mathbb{R}^{B \times N}.
$$

**Sanity checks that must hold in any correct implementation.**

1. $\sum_k \mathbf{G}_{b,k} = 0$ for every sample, because
   $\sum_k \mathbf{P}_{b,k} = \sum_k \mathbf{Y}_{b,k} = 1$. A non-zero row sum
   means the shift or the normalisation is wrong.
2. $\mathbf{G}_{b,k} < 0$ for the true class and $> 0$ for the others, so SGD
   pushes the true logit up and the others down.
3. Every entry satisfies $|\mathbf{G}_{b,k}| \le 1/B$.

This combined form is why the softmax layer has no separate backward method in
many implementations: the loss supplies $\mathbf{G}$ directly and the softmax
backward alone (multiplying by the Jacobian) is never needed. It is also
numerically better, because it never divides by a tiny probability.

---

## 15. Mini-batch gradient averaging

All gradients above are already batch-averaged: the $1/B$ from the loss appears
once, at the softmax stage, and every downstream layer propagates it linearly.
The convolution and dense parameter gradients therefore read

$$
\nabla_{\mathbf{W}} L \;=\; \frac{1}{B}\sum_{b=0}^{B-1} \nabla_{\mathbf{W}} L_b ,
$$

i.e. an **average, not a sum**, of the per-sample gradients. Two consequences
worth stating explicitly in a viva:

- **Equivalence.** Averaging over a batch of 32 is the same estimator as summing
  per-sample gradients and dividing by 32. The implementation may compute either
  as long as the $1/B$ appears exactly once.
- **No double scaling.** A frequent bug is dividing by $B$ both in the loss
  gradient and again in the parameter update; the effective learning rate then
  shrinks by $32\times$ and the model appears not to learn at all.

With $B = 32$ and the Phase 1 training subset of 10,000 images an epoch is
$\lceil 10000/32 \rceil = 313$ mini-batches (the last one holding 16 samples).
If the final partial batch is averaged with its own size, the estimator stays
unbiased; the project will always divide by the actual batch length.

---

## 16. SGD parameter updates

Plain stochastic gradient descent, with learning rate $\eta$:

$$
\theta \;\leftarrow\; \theta \;-\; \eta \,\nabla_\theta L,
\qquad
\theta \in \{\mathbf{W}_1, \boldsymbol{\beta}_1, \mathbf{W}_2, \boldsymbol{\beta}_2\}.
$$

Per parameter that is

$$
\mathbf{W}_1 \leftarrow \mathbf{W}_1 - \eta \nabla_{\mathbf{W}_1} L, \quad
\boldsymbol{\beta}_1 \leftarrow \boldsymbol{\beta}_1 - \eta \nabla_{\boldsymbol{\beta}_1} L, \quad
\mathbf{W}_2 \leftarrow \mathbf{W}_2 - \eta \nabla_{\mathbf{W}_2} L, \quad
\boldsymbol{\beta}_2 \leftarrow \boldsymbol{\beta}_2 - \eta \nabla_{\boldsymbol{\beta}_2} L .
$$

The frozen configuration sets $\eta = 0.01$ and no momentum, no weight decay, no
learning-rate schedule, so "SGD" here means the plain update written above. Keras'
`SGD(learning_rate=0.01)` with `momentum=0.0` is the same rule; any momentum or
decay added later must be added to **both** implementations and recorded in
`configs/experiment_config.json`.

**Update order.** All parameter gradients are computed from the same forward pass
before any parameter is modified (a "simultaneous" update). Updating $\mathbf{W}_1$
before computing $\nabla_{\mathbf{W}_2}$ would make the two gradients belong to
different functions and is not SGD.

**Why 0.01.** Small enough that the mean cross-entropy decreases monotonically on
a 32-sample batch, large enough to make progress within a handful of epochs on
CPU. Whether it is a good choice is an empirical question; the loss curve in
Phase 4 will answer it, and no claim is made here.

---

## 17. Parameter counting

| Layer | Formula | Substitution | Parameters |
|---|---|---|---|
| Conv2D | $(K \cdot K \cdot C + 1)\cdot F$ | $(3\cdot3\cdot1 + 1)\cdot 8 = 10 \cdot 8$ | **80** |
| ReLU | — | — | 0 |
| MaxPool2D | — | — | 0 |
| Flatten | — | — | 0 |
| Dense | $M\cdot N + N$ | $1352\cdot10 + 10 = 13{,}520 + 10$ | **13,530** |
| Softmax | — | — | 0 |
| **Total** | | $80 + 13{,}530$ | **13,610** |

Notes on the derivation:

- **Convolution.** Each of the $F = 8$ filters owns a $3\times3\times1$ kernel (9
  weights) and one bias, so $10$ per filter and $80$ in total. Bias included.
- **Dense.** $M = 1352$ inputs reach each of the $N = 10$ outputs: 13,520 weights.
  Each output has one bias: 10 more. Bias included.
- **No trainable parameters** in ReLU, max pooling, flatten or softmax: they are
  fixed functions with no free values.
- The figures are computed by
  `describe_architecture()` in `src/cnn_scratch/models/architecture.py` and
  asserted in `tests/test_architecture.py`, so the document cannot drift from
  the configuration.

---

## 18. He initialisation

Weights are drawn from a zero-mean Gaussian with variance scaled by the number of
inputs to the unit (He et al., 2015):

$$
\mathbf{W} \sim \mathcal{N}\!\left(0,\; \frac{2}{n_{\text{in}}}\right),
\qquad
\sigma_{\mathbf{W}} = \sqrt{\frac{2}{n_{\text{in}}}},
$$

where $n_{\text{in}}$ is the fan-in of one output unit. For the frozen
architecture:

| Layer | $n_{\text{in}}$ | $\sigma$ | Value |
|---|---|---|---|
| Conv2D | $K \cdot K \cdot C = 9$ | $\sqrt{2/9}$ | $\approx 0.4714$ |
| Dense | $M = 1352$ | $\sqrt{2/1352}$ | $\approx 0.0385$ |

Biases start at **zero**, which is safe because the random weights already break
the symmetry between units.

**Why $2/n_{\text{in}}$.** ReLU zeroes half its inputs, so the variance of the
activations halves at every layer; the factor 2 compensates and keeps the signal
variance roughly constant through the stack. Xavier/Glorot's $1/n_{\text{in}}$
assumes a symmetric activation and would shrink the signal here.

**Exactly one initialisation.** The initial weights must be generated once, from
the seeded generator (`numpy.random.default_rng(42)`), and reused by the Keras
model rather than re-drawn — otherwise the two implementations start from
different points and the comparison in Phase 6 measures the initialisation rather
than the implementations. Keras' `he_normal` uses the same $2/n_{\text{in}}$
variance, so transferring the NumPy-drawn arrays into the Keras layers is the
cleanest way to guarantee identical starting parameters.

---

## 19. Worked numerical examples

Every example below is small enough to verify by hand. Arithmetic is exact to
the precision shown; intermediate values are rounded for display only.

### 19.1 Convolution (forward)

One channel, `valid` padding, stride 1, $K = 3$, bias $\beta = 0$:

```text
X (4x4)                     W (3x3)
1 2 0 1                     1  0 -1
0 1 3 2                     1  0 -1
2 1 0 1                     1  0 -1
1 0 2 1
```

The output is $2\times2$. For $i=0, j=0$ multiply the top-left $3\times3$ patch
element-wise with $W$ and sum:

```text
patch           element-wise products      sum
1 2 0           1*1  2*0  0*(-1)  =  1
0 1 3           0*1  1*0  3*(-1)  = -3
2 1 0           2*1  1*0  0*(-1)  =  2
                                        ----
                                         0.0
```

Repeating for the other three positions:

```text
Y =  [ 0.0   0.0 ]
     [-2.0  -2.0 ]
```

Check for $i=1,j=0$: the patch is rows 1-3, columns 0-2,
$\;(0\cdot1 + 1\cdot0 + 3\cdot(-1)) + (2\cdot1 + 1\cdot0 + 0\cdot(-1)) + (1\cdot1 + 0\cdot0 + 2\cdot(-1)) = -3 + 2 - 1 = -2$. ✓

Adding a bias $\beta = 1$ would give $\mathbf{Z} = \mathbf{Y} + 1 = [[1,1],[1,1]]$.
The kernel here is an edge detector: it responds to a vertical gradient.

### 19.2 Convolution (backward, $dW$ accumulation)

Take the same $\mathbf{X}$ and the upstream gradient
$\partial L/\partial \mathbf{Y} = \mathbf{1}_{2\times2}$ (every output
contributes equally). Because the four windows overlap, each input pixel is
touched by up to four windows and the filter gradient is the **sum** of all its
contributions:

$$
\frac{\partial L}{\partial W_{u,v}} = \sum_{i=0}^{1}\sum_{j=0}^{1}
\frac{\partial L}{\partial Y_{i,j}}\; X_{i+u,\,j+v}
= \sum_{i,j} X_{i+u,\,j+v}.
$$

```text
dW[0,0] = X[0,0]+X[0,1]+X[1,0]+X[1,1] = 1+2+0+1 = 4
dW[0,1] = X[0,1]+X[0,2]+X[1,1]+X[1,2] = 2+0+1+3 = 6
dW[0,2] = X[0,2]+X[0,3]+X[1,2]+X[1,3] = 0+1+3+2 = 6
dW[1,0] = X[1,0]+X[1,1]+X[2,0]+X[2,1] = 0+1+2+1 = 4
dW[1,1] = X[1,1]+X[1,2]+X[2,1]+X[2,2] = 1+3+1+0 = 5
dW[1,2] = X[1,2]+X[1,3]+X[2,2]+X[2,3] = 3+2+0+1 = 6
dW[2,0] = X[2,0]+X[2,1]+X[3,0]+X[3,1] = 2+1+1+0 = 4
dW[2,1] = X[2,1]+X[2,2]+X[3,1]+X[3,2] = 1+0+0+2 = 3
dW[2,2] = X[2,2]+X[2,3]+X[3,2]+X[3,3] = 0+1+2+1 = 4
```

```text
dW =  [ 4  6  6 ]
      [ 4  5  6 ]
      [ 4  3  4 ]          db = sum(dY) = 4
```

Cross-check: summing the nine entries counts every input pixel once per window
that covers it, so the total equals the sum of the four window sums,
$10 + 11 + 10 + 11 = 42$. ✓

The important observation is that **overlapping receptive fields cause
accumulation**: the single pixel $X[1,1]$ contributes once through each of the
four windows that contain it, landing in $dW[1,1]$, $dW[1,0]$, $dW[0,1]$ and
$dW[0,0]$ respectively. An implementation that assigns (`=`) instead of
accumulating (`+=`) would silently keep only the last window's contribution.

### 19.3 ReLU (forward and backward)

```text
z        = [-2.0, -0.5,  0.0,  1.5,  3.0]
a = max(0,z) = [ 0.0,  0.0,  0.0,  1.5,  3.0]

upstream dL/da = [ 1.0,  2.0,  3.0,  4.0,  5.0]
mask (z > 0)   = [  F ,   F ,   F ,   T ,   T ]
dL/dz          = [ 0.0,  0.0,  0.0,  4.0,  5.0]
```

The negative inputs pass no gradient; $z=0$ uses the subgradient 0, so the third
entry is also 0.

### 19.4 Max pooling (forward and backward)

$4\times4$ input, $2\times2$ windows, stride 2:

```text
X                        windows / argmax              Y
1 3 2 4                  {1,3,5,6} -> 6 at (1,1)       6 4
5 6 1 0                  {2,4,1,0} -> 4 at (0,1)       2 9
0 2 7 3                  {0,2,1,1} -> 2 at (0,1)
1 1 2 9                  {7,3,2,9} -> 9 at (1,1)
```

With upstream $\partial L/\partial \mathbf{Y} = [[1, 2], [3, 4]]$ the gradient is
routed only to the selected positions, laid out like $X$:

```text
dL/dX =
0  0  0  2        X[0,3] = 4, max of the top-right window    -> gets 2
0  1  0  0        X[1,1] = 6, max of the top-left window     -> gets 1
0  3  0  0        X[2,1] = 2, max of the bottom-left window   -> gets 3
0  0  0  4        X[3,3] = 9, max of the bottom-right window  -> gets 4
```

Sum check: $\sum \partial L/\partial X = 1+2+3+4 = 10 = \sum \partial L/\partial Y$. ✓
Every non-selected value, including the large $X[2,2] = 7$, receives exactly 0 —
being large is not enough, it must be the maximum of its window.

### 19.5 Dense layer gradients

One sample, $M = 3$ inputs, $N = 2$ outputs ($W \in \mathbb{R}^{3 \times 2}$):

```text
a = [1.0, 2.0, 0.5]  # M=3 inputs (row vector)
W = [[ 0.0, -0.5],      # W shape (3,2): input 1's weights
     [-0.25,  2.75],   # input 2's weights
     [ 0.0, -6.5]]    # input 3's weights
b = [0.1, -0.2]

# Forward: z_n = \sum_m a_m * W_{m,n} + b_n
z_1 = 1.0*0 + 2.0*(-0.25) + 0.5*0 + 0.1 = -0.4
z_2 = 1.0*(-0.5) + 2.0*2.75 + 0.5*(-6.5) + (-0.2) = 1.55
z = [-0.4, 1.55]
```

With upstream gradient $\mathbf{g} = \partial L/\partial z = [0.3, -0.1]$ (shape $(N,) = (2,)$):

Correct gradients (using the convention $W \in \mathbb{R}^{M \times N}$,
$dW_{m,n} = a_m * g_n$, $da_m = \sum_k g_k * W_{m,k}$):

```text
dW = a^T g = [[ 0.30, -0.10],
                [ 0.60, -0.20],
                [ 0.15, -0.05]]  # shape (M,N) = (3,2)
db = [0.3, -0.1]  # shape (N,) = (2,)
da = g @ W^T = [0.05, -0.35, 0.65]  # shape (M,) = (3,)
```

Shapes summary:
- $W$: $(3,2)$ — input-by-output matrix
- $dW$: $(3,2)$ — same shape as $W$
- $b$: $(2,)$ — bias per output
- $db$: $(2,)$ — same shape as $b$  
- $a$: $(3,)$ — input vector
- $da$: $(3,)$ — same shape as $a$

For a batch, `dW = A^T G` accumulates the outer products of all samples and
`db = sum(G, axis=0)`; with the $1/B$ already inside $G$ both are averages.

### 19.6 Softmax and cross-entropy

Logits $z = [2.0, 1.0, 0.1]$, true class 0 ($\mathbf{y} = [1,0,0]$). The maximum
is $m = 2.0$:

```text
shifted       z - m           = [ 0.0, -1.0, -1.9]
exp                e^(z-m)    = [ 1.000000, 0.367879, 0.149569]
sum                            1.517448
p = exp / sum                 = [ 0.659001, 0.242433, 0.098566]      (sums to 1.000000)

loss  = -log(0.659001)        = 0.41703
```

Gradient with respect to the logits (single sample, so $B = 1$):

```text
p - y = [ 0.659001 - 1,  0.242433 - 0,  0.098566 - 0 ]
      = [-0.340999,      0.242433,      0.098566]           row sum = 0 ✓
```

For a batch of 32 the same expression is divided by 32, giving entries of at most
$1/32 = 0.03125$ in magnitude.

### 19.7 SGD update

Parameter $W_{0,0} = 0.5$ with gradient $0.3$, bias $b_0 = 0.1$ with gradient
$0.3$, learning rate $\eta = 0.01$:

$$
W_{0,0} \leftarrow 0.5 - 0.01 \times 0.3 = 0.497,\qquad
b_0 \leftarrow 0.1 - 0.01 \times 0.3 = 0.097 .
$$

Both move _against_ the gradient; a gradient of $+0.3$ pulls the parameter down
by $0.003$. The bias in the example of 19.5 (`b[1]` with gradient $-0.1$) would
move up: $b_1 = -0.2 - 0.01 \times (-0.1) = -0.199$.

---

## 20. References

- He, K., Zhang, X., Ren, S. & Sun, J. (2015). *Delving Deep into Rectifiers:
  Surpassing Human-Level Performance on ImageNet Classification.* ICCV.
- LeCun, Y., Bottou, L., Bengio, Y. & Haffner, P. (1998). *Gradient-Based
  Learning Applied to Document Recognition.* Proceedings of the IEEE 86(11).
- Goodfellow, I., Bengio, Y. & Courville, A. (2016). *Deep Learning.* MIT Press.
  Chapters 6, 9 and 11.
- Nielsen, M. (2015). *Neural Networks and Deep Learning*, chapter 6
  (conventions for convolution and pooling).
- Dumoulin, V. & Visin, F. (2016). *A Guide to Convolution Arithmetic for Deep
  Learning.* arXiv:1603.07285 (padding, stride and output-shape formulas).
- Xiao, H., Rasul, K. & Vollgraf, R. (2017). *Fashion-MNIST: a Novel Image
  Dataset for Benchmarking Machine Learning Algorithms.*
- TensorFlow/Keras API documentation for `Conv2D`, `MaxPooling2D`, `Dense`,
  `categorical_crossentropy` and `SGD`.

