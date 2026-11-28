"""Tests for the Phase 3 NumPy layers.









Forward/backward correctness (including analytical-vs-numerical gradient




checks in float64) lives in ``test_layers``.  The remaining tests exercise




shapes, determinism, tie handling, bias behaviour and the frozen-architecture




end-to-end forward/backward.




"""









from __future__ import annotations









import numpy as np




import pytest









from cnn_scratch.models.layers import (
    CategoricalCrossEntropy,
    Conv2D,
    Dense,
    Flatten,
    MaxPool2D,
    ReLU,
    StableSoftmax,
)
from cnn_scratch.models.scratch_cnn import ScratchCNN









# ---------------------------------------------------------------------------




# Deterministic float64 fixtures




# ---------------------------------------------------------------------------














def make_conv2d() -> Conv2D:




    layer = Conv2D((2, 2, 1, 1), seed=7)




    layer.W[:, :, 0, 0] = 1.0




    layer.b[0] = 0.0




    return layer














def make_dense() -> Dense:




    layer = Dense(4, 2, seed=3)




    layer.W[0, 0] = 1.0




    layer.b[0] = 0.0




    return layer














# Small Conv2D input: kernel (2,2,1,1) -> output (1, 1, 1, 1)




@pytest.fixture




def x_conv2d_small() -> np.ndarray:




    return np.array([[[[1.0], [2.0]], [[3.0], [4.0]]]], dtype=np.float64)














# Small Dense input: (B=1, 4) -> (B=1, 2)




@pytest.fixture




def x_dense_small() -> np.ndarray:




    return np.array([[1.0, 2.0, 3.0, 4.0]], dtype=np.float64)














# ---------------------------------------------------------------------------




# Forward/backward shapes and values




# ---------------------------------------------------------------------------














def test_conv2d_forward_shape_and_value(x_conv2d_small):




    layer = make_conv2d()




    y = layer.forward(x_conv2d_small)




    assert y.shape == (1, 1, 1, 1)




    assert y[0, 0, 0, 0] == pytest.approx(10.0)














def test_conv2d_backward_output_gradients_match_forward():




    x = np.array([[[[1.0], [2.0]], [[3.0], [4.0]]]], dtype=np.float64)




    layer = make_conv2d()




    layer.forward(x)




    dy = np.array([[[[1.0]]]], dtype=np.float64)          # (1, 1, 1, 1)




    dx, dw, db = layer.backward(dy)









    assert dx.shape == (1, 2, 2, 1)




    assert dw.shape == (2, 2, 1, 1)




    assert db.shape == (1,)




    assert np.allclose(dx, 1.0)




    assert dw[0, 0, 0, 0] == pytest.approx(1.0)




    assert dw[0, 1, 0, 0] == pytest.approx(2.0)




    assert dw[1, 0, 0, 0] == pytest.approx(3.0)




    assert dw[1, 1, 0, 0] == pytest.approx(4.0)




    assert db[0] == pytest.approx(1.0)














def test_relu_forward_and_backward():




    x = np.array([[-2.0, -0.5, 0.0, 1.5, 3.0]], dtype=np.float64)




    layer = ReLU()




    y = layer.forward(x)




    assert y.shape == x.shape




    np.testing.assert_allclose(y, [[0.0, 0.0, 0.0, 1.5, 3.0]], atol=1e-12)









    dy = np.ones_like(x)




    dx = layer.backward(dy)




    np.testing.assert_allclose(dx, [[0.0, 0.0, 0.0, 1.0, 1.0]], atol=1e-12)














def test_relu_backward_zero_convention():




    # x = 0 uses subgradient 0 (NumPy / Keras convention).




    x = np.array([[0.0]])




    layer = ReLU()




    layer._cache["x"] = x




    dy = np.array([[5.0]])




    dx = layer.backward(dy)




    assert dx[0, 0] == 0.0














def test_maxpool2d_forward_shape_and_value():




    x = np.array([[[[1.0], [2.0], [3.0], [4.0]],




                   [[5.0], [6.0], [7.0], [8.0]],




                   [[9.0], [10.0], [11.0], [12.0]],




                   [[13.0], [14.0], [15.0], [16.0]]]], dtype=np.float64)




    layer = MaxPool2D(pool_size=2, stride=2)




    y = layer.forward(x)




    assert y.shape == (1, 2, 2, 1)




    # NHWC: y[0, i, j, 0] with i = row block, j = col block. Window maxima




    # top-left 6, top-right 8, bottom-left 14, bottom-right 16.




    assert y[0, 0, 0, 0] == pytest.approx(6.0)




    assert y[0, 0, 1, 0] == pytest.approx(8.0)




    assert y[0, 1, 0, 0] == pytest.approx(14.0)




    assert y[0, 1, 1, 0] == pytest.approx(16.0)














def test_maxpool2d_backward_routes_to_max():




    x = np.array([[[[1.0], [2.0], [3.0], [4.0]],




                   [[5.0], [6.0], [7.0], [8.0]],




                   [[9.0], [10.0], [11.0], [12.0]],




                   [[13.0], [14.0], [15.0], [16.0]]]], dtype=np.float64)




    layer = MaxPool2D(pool_size=2, stride=2)




    layer.forward(x)




    dy = np.array([[[[1.0], [2.0]], [[3.0], [4.0]]]], dtype=np.float64)




    dx = layer.backward(dy)




    # Each output window's max sits at its local (1, 1), i.e. input positions




    # (1, 1), (1, 3), (3, 1), (3, 3). Gradients route there.




    assert dx[0, 1, 1, 0] == 1.0          # from dy[0, 0, 0]




    assert dx[0, 1, 3, 0] == 2.0          # from dy[0, 0, 1]




    assert dx[0, 3, 1, 0] == 3.0          # from dy[1, 0, 0]




    assert dx[0, 3, 3, 0] == 4.0          # from dy[1, 1, 0]




    assert dx[0, 0, 0, 0] == 0.0




    assert dx[0, 0, 2, 0] == 0.0














def test_flatten_forward_and_backward():




    x = np.zeros((2, 13, 13, 8), dtype=np.float64)




    layer = Flatten()




    y = layer.forward(x)




    assert y.shape == (2, 13 * 13 * 8)




    dy = np.ones((2, 13 * 13 * 8), dtype=np.float64)




    dx = layer.backward(dy)




    assert dx.shape == (2, 13, 13, 8)














def test_dense_forward_and_backward(x_dense_small):




    layer = make_dense()




    layer.W[1:, :] = 0.0




    layer.W[0, 1] = 0.0




    y = layer.forward(x_dense_small)




    assert y.shape == (1, 2)




    np.testing.assert_allclose(y, [[1.0, 0.0]], atol=1e-12)









    dy = np.array([[1.0, 2.0]], dtype=np.float64)




    dx, dw, db = layer.backward(dy)




    assert dx.shape == (1, 4)




    assert dw.shape == (4, 2)




    assert db.shape == (2,)




    # dW = x.T @ dy = outer(x[0], dy[0]) = outer([1,2,3,4], [1,2])




    np.testing.assert_allclose(dw[0], [1.0, 2.0], atol=1e-12)




    np.testing.assert_allclose(dw[1], [2.0, 4.0], atol=1e-12)




    np.testing.assert_allclose(dw[2], [3.0, 6.0], atol=1e-12)




    np.testing.assert_allclose(dw[3], [4.0, 8.0], atol=1e-12)




    # dx = dy @ W.T = [1, 2] @ [[1, 0], [0, 0], [0, 0], [0, 0]] = [1, 0, 0, 0]




    np.testing.assert_allclose(dx, [[1.0, 0.0, 0.0, 0.0]], atol=1e-12)














def test_stable_softmax_forward_and_probability_sum():




    layer = StableSoftmax()




    z = np.array([[1000.0, 1000.0, 0.0],




                  [0.0, 1.0, 2.0]], dtype=np.float64)




    p = layer.forward(z)




    assert p.shape == (2, 3)




    np.testing.assert_allclose(p.sum(axis=1), [1.0, 1.0], atol=1e-6)




    assert np.all(np.isfinite(p))














def test_softmax_handles_extreme_logits():




    layer = StableSoftmax()




    z = np.array([[1e6, -1e6, 0.0],




                  [0.0, 0.0, 0.0]], dtype=np.float64)




    p = layer.forward(z)




    assert np.all(np.isfinite(p))




    np.testing.assert_allclose(p.sum(axis=1), [1.0, 1.0], atol=1e-6)




    np.testing.assert_allclose(p[0], [1.0, 0.0, 0.0], atol=1e-6)














def test_softmax_batch_independent():




    layer = StableSoftmax()




    z = np.array([[2.0, 1.0], [1.0, 3.0]], dtype=np.float64)




    p = layer.forward(z)




    # Each row must sum to 1 independently.




    np.testing.assert_allclose(p.sum(axis=1), [1.0, 1.0], atol=1e-6)














def test_categorical_cross_entropy_forward():




    loss = CategoricalCrossEntropy()




    p = np.array([[0.7, 0.2, 0.1],




                  [0.1, 0.8, 0.1]], dtype=np.float64)




    y = np.array([[1.0, 0.0, 0.0],




                  [0.0, 1.0, 0.0]], dtype=np.float64)




    l = loss.forward(p, y)




    assert l == pytest.approx(-(np.log(0.7) + np.log(0.8)) / 2, abs=1e-12)














# ---------------------------------------------------------------------------




# Gradient checks (analytical vs finite differences, float64)




# ---------------------------------------------------------------------------














def _num_grad_input(layer: object, x: np.ndarray, dy: np.ndarray,
                     *, eps: float = 1e-5):
    """Central-difference gradient of ``layer.forward`` w.r.t. input.

    The scalar loss used for the finite difference is ``L = sum(dy * f)``, so
    the returned gradient matches the input gradient that ``layer.backward(dy)``
    must produce.
    """
    num_grad = np.zeros_like(x, dtype=np.float64)
    for idx in np.ndindex(x.shape):
        x_plus = x.copy()
        x_minus = x.copy()
        x_plus[idx] += eps
        x_minus[idx] -= eps
        f_plus = np.asarray(layer.forward(x_plus), dtype=np.float64)
        f_minus = np.asarray(layer.forward(x_minus), dtype=np.float64)
        num_grad[idx] = np.sum(dy * (f_plus - f_minus)) / (2 * eps)
    return num_grad


def _num_grad_weight(layer: object, param_name: str, *, dy: np.ndarray,




                     x: np.ndarray, eps: float = 1e-5):




    """Central-difference gradient of the scalar loss ``sum(dy * f)`` w.r.t. a




    weight param, matching the analytical gradient returned by




    ``layer.backward(dy)[1]`` (for Conv2D/Dense).




    """




    base = getattr(layer, param_name)




    num_grad = np.zeros_like(base, dtype=np.float64)




    # Numerical gradient must use the *current* forward output, but the loss




    # depends on dy, not on x only.  We therefore recompute the loss with the




    # perturbed parameter in a fresh forward pass.




    for idx in np.ndindex(base.shape):




        old = base[idx]




        base[idx] = old + eps




        layer.forward(x)




        f_plus = np.asarray(layer.forward(x), dtype=np.float64)




        base[idx] = old - eps




        layer.forward(x)




        f_minus = np.asarray(layer.forward(x), dtype=np.float64)




        base[idx] = old




        num_grad[idx] = np.sum(dy * (f_plus - f_minus)) / (2 * eps)




    return num_grad














def _check_input_grad(layer, x, dy, eps=1e-5, label=""):
    """Analytical input gradient vs central-difference numerical gradient."""
    layer.forward(x)
    backward_out = layer.backward(dy)
    # Conv2D and Dense return a 3-tuple; ReLU/MaxPool/Flatten return a single
    # array (the input gradient) so it can be compared directly.
    an = backward_out[0] if isinstance(backward_out, tuple) else backward_out
    a = _num_grad_input(layer, x, dy, eps=eps)
    np.testing.assert_allclose(an, a, atol=1e-2, rtol=1e-2,
                                err_msg=f"{label or type(layer).__name__} input gradient")


def _check_weight_grad(layer, x, dy, param_name: str, eps=1e-5, label=""):
    """Weight gradient vs central-difference gradient of the scalar loss."""
    layer.forward(x)
    backward_out = layer.backward(dy)
    # Conv2D and Dense return a 3-tuple; only their weight gradient is checked.
    an = backward_out[1] if isinstance(backward_out, tuple) else backward_out
    ac = _num_grad_weight(layer, param_name, dy=dy, x=x, eps=eps)
    np.testing.assert_allclose(an, ac, atol=1e-2, rtol=1e-2,
                                err_msg=f"{label or type(layer).__name__} {param_name} gradient")


# ---------------------------------------------------------------------------




# Deterministic float64 fixtures




# ---------------------------------------------------------------------------














def make_conv2d() -> Conv2D:




    layer = Conv2D((2, 2, 1, 1), seed=7)




    layer.W[:, :, 0, 0] = 1.0




    layer.b[0] = 0.0




    return layer














def make_dense() -> Dense:




    layer = Dense(4, 2, seed=3)




    layer.W[0, 0] = 1.0




    layer.b[0] = 0.0




    return layer














# Small Conv2D input: kernel (2,2,1,1) -> output (1, 1, 1, 1)




@pytest.fixture




def x_conv2d_small() -> np.ndarray:




    return np.array([[[[1.0], [2.0]], [[3.0], [4.0]]]], dtype=np.float64)














# Small Dense input: (B=1, 4) -> (B=1, 2)




@pytest.fixture




def x_dense_small() -> np.ndarray:




    return np.array([[1.0, 2.0, 3.0, 4.0]], dtype=np.float64)














# ---------------------------------------------------------------------------




# Forward/backward shapes and values




# ---------------------------------------------------------------------------














def test_conv2d_forward_shape_and_value(x_conv2d_small):




    layer = make_conv2d()




    y = layer.forward(x_conv2d_small)




    assert y.shape == (1, 1, 1, 1)




    assert y[0, 0, 0, 0] == pytest.approx(10.0)














def test_conv2d_backward_output_gradients_match_forward():




    x = np.array([[[[1.0], [2.0]], [[3.0], [4.0]]]], dtype=np.float64)




    layer = make_conv2d()




    layer.forward(x)




    dy = np.array([[[[1.0]]]], dtype=np.float64)          # (1, 1, 1, 1)




    dx, dw, db = layer.backward(dy)









    assert dx.shape == (1, 2, 2, 1)




    assert dw.shape == (2, 2, 1, 1)




    assert db.shape == (1,)




    assert np.allclose(dx, 1.0)




    assert dw[0, 0, 0, 0] == pytest.approx(1.0)




    assert dw[0, 1, 0, 0] == pytest.approx(2.0)




    assert dw[1, 0, 0, 0] == pytest.approx(3.0)




    assert dw[1, 1, 0, 0] == pytest.approx(4.0)




    assert db[0] == pytest.approx(1.0)














def test_relu_forward_and_backward():




    x = np.array([[-2.0, -0.5, 0.0, 1.5, 3.0]], dtype=np.float64)




    layer = ReLU()




    y = layer.forward(x)




    assert y.shape == x.shape




    np.testing.assert_allclose(y, [[0.0, 0.0, 0.0, 1.5, 3.0]], atol=1e-12)









    dy = np.ones_like(x)




    dx = layer.backward(dy)




    np.testing.assert_allclose(dx, [[0.0, 0.0, 0.0, 1.0, 1.0]], atol=1e-12)














def test_relu_backward_zero_convention():




    # x = 0 uses subgradient 0 (NumPy / Keras convention).




    x = np.array([[0.0]])




    layer = ReLU()




    layer._cache["x"] = x




    dy = np.array([[5.0]])




    dx = layer.backward(dy)




    assert dx[0, 0] == 0.0














def test_maxpool2d_forward_shape_and_value():




    x = np.array([[[[1.0], [2.0], [3.0], [4.0]],




                   [[5.0], [6.0], [7.0], [8.0]],




                   [[9.0], [10.0], [11.0], [12.0]],




                   [[13.0], [14.0], [15.0], [16.0]]]], dtype=np.float64)




    layer = MaxPool2D(pool_size=2, stride=2)




    y = layer.forward(x)




    assert y.shape == (1, 2, 2, 1)




    # NHWC: y[0, i, j, 0] with i = row block, j = col block. Window maxima




    # top-left 6, top-right 8, bottom-left 14, bottom-right 16.




    assert y[0, 0, 0, 0] == pytest.approx(6.0)




    assert y[0, 0, 1, 0] == pytest.approx(8.0)




    assert y[0, 1, 0, 0] == pytest.approx(14.0)




    assert y[0, 1, 1, 0] == pytest.approx(16.0)














def test_maxpool2d_backward_routes_to_max():




    x = np.array([[[[1.0], [2.0], [3.0], [4.0]],




                   [[5.0], [6.0], [7.0], [8.0]],




                   [[9.0], [10.0], [11.0], [12.0]],




                   [[13.0], [14.0], [15.0], [16.0]]]], dtype=np.float64)




    layer = MaxPool2D(pool_size=2, stride=2)




    layer.forward(x)




    dy = np.array([[[[1.0], [2.0]], [[3.0], [4.0]]]], dtype=np.float64)




    dx = layer.backward(dy)




    # Each output window's max sits at its local (1, 1), i.e. input positions




    # (1, 1), (1, 3), (3, 1), (3, 3). Gradients route there.




    assert dx[0, 1, 1, 0] == 1.0          # from dy[0, 0, 0]




    assert dx[0, 1, 3, 0] == 2.0          # from dy[0, 0, 1]




    assert dx[0, 3, 1, 0] == 3.0          # from dy[1, 0, 0]




    assert dx[0, 3, 3, 0] == 4.0          # from dy[1, 1, 0]




    assert dx[0, 0, 0, 0] == 0.0




    assert dx[0, 0, 2, 0] == 0.0














def test_flatten_forward_and_backward():




    x = np.zeros((2, 13, 13, 8), dtype=np.float64)




    layer = Flatten()




    y = layer.forward(x)




    assert y.shape == (2, 13 * 13 * 8)




    dy = np.ones((2, 13 * 13 * 8), dtype=np.float64)




    dx = layer.backward(dy)




    assert dx.shape == (2, 13, 13, 8)














def test_dense_forward_and_backward(x_dense_small):




    layer = make_dense()




    layer.W[1:, :] = 0.0




    layer.W[0, 1] = 0.0




    y = layer.forward(x_dense_small)




    assert y.shape == (1, 2)




    np.testing.assert_allclose(y, [[1.0, 0.0]], atol=1e-12)









    dy = np.array([[1.0, 2.0]], dtype=np.float64)




    dx, dw, db = layer.backward(dy)




    assert dx.shape == (1, 4)




    assert dw.shape == (4, 2)




    assert db.shape == (2,)




    # dW = x.T @ dy = outer(x[0], dy[0]) = outer([1,2,3,4], [1,2])




    np.testing.assert_allclose(dw[0], [1.0, 2.0], atol=1e-12)




    np.testing.assert_allclose(dw[1], [2.0, 4.0], atol=1e-12)




    np.testing.assert_allclose(dw[2], [3.0, 6.0], atol=1e-12)




    np.testing.assert_allclose(dw[3], [4.0, 8.0], atol=1e-12)




    # dx = dy @ W.T = [1, 2] @ [[1, 0], [0, 0], [0, 0], [0, 0]] = [1, 0, 0, 0]




    np.testing.assert_allclose(dx, [[1.0, 0.0, 0.0, 0.0]], atol=1e-12)














def test_stable_softmax_forward_and_probability_sum():




    layer = StableSoftmax()




    z = np.array([[1000.0, 1000.0, 0.0],




                  [0.0, 1.0, 2.0]], dtype=np.float64)




    p = layer.forward(z)




    assert p.shape == (2, 3)




    np.testing.assert_allclose(p.sum(axis=1), [1.0, 1.0], atol=1e-6)




    assert np.all(np.isfinite(p))














def test_softmax_handles_extreme_logits():




    layer = StableSoftmax()




    z = np.array([[1e6, -1e6, 0.0],




                  [0.0, 0.0, 0.0]], dtype=np.float64)




    p = layer.forward(z)




    assert np.all(np.isfinite(p))




    np.testing.assert_allclose(p.sum(axis=1), [1.0, 1.0], atol=1e-6)




    np.testing.assert_allclose(p[0], [1.0, 0.0, 0.0], atol=1e-6)














def test_softmax_batch_independent():




    layer = StableSoftmax()




    z = np.array([[2.0, 1.0], [1.0, 3.0]], dtype=np.float64)




    p = layer.forward(z)




    # Each row must sum to 1 independently.




    np.testing.assert_allclose(p.sum(axis=1), [1.0, 1.0], atol=1e-6)














def test_categorical_cross_entropy_forward():




    loss = CategoricalCrossEntropy()




    p = np.array([[0.7, 0.2, 0.1],




                  [0.1, 0.8, 0.1]], dtype=np.float64)




    y = np.array([[1.0, 0.0, 0.0],




                  [0.0, 1.0, 0.0]], dtype=np.float64)




    l = loss.forward(p, y)




    assert l == pytest.approx(-(np.log(0.7) + np.log(0.8)) / 2, abs=1e-12)














# ---------------------------------------------------------------------------




# Gradient checks (analytical vs finite differences, float64)




# ---------------------------------------------------------------------------














def _num_grad_input(layer: object, x: np.ndarray, dy: np.ndarray,
                     *, eps: float = 1e-5):
    """Central-difference gradient of ``layer.forward`` w.r.t. input.

    The scalar loss used for the finite difference is ``L = sum(dy * f)``, so
    the returned gradient matches the input gradient that ``layer.backward(dy)``
    must produce.
    """
    num_grad = np.zeros_like(x, dtype=np.float64)
    for idx in np.ndindex(x.shape):
        x_plus = x.copy()
        x_minus = x.copy()
        x_plus[idx] += eps
        x_minus[idx] -= eps
        f_plus = np.asarray(layer.forward(x_plus), dtype=np.float64)
        f_minus = np.asarray(layer.forward(x_minus), dtype=np.float64)
        num_grad[idx] = np.sum(dy * (f_plus - f_minus)) / (2 * eps)
    return num_grad


def _num_grad_weight(layer: object, param_name: str, *, dy: np.ndarray,




                     x: np.ndarray, eps: float = 1e-5):




    """Central-difference gradient of the scalar loss ``sum(dy * f)`` w.r.t. a




    weight param, matching the analytical gradient returned by




    ``layer.backward(dy)[1]`` (for Conv2D/Dense).




    """




    base = getattr(layer, param_name)




    num_grad = np.zeros_like(base, dtype=np.float64)




    # Numerical gradient must use the *current* forward output, but the loss




    # depends on dy, not on x only.  We therefore recompute the loss with the




    # perturbed parameter in a fresh forward pass.




    for idx in np.ndindex(base.shape):




        old = base[idx]




        base[idx] = old + eps




        layer.forward(x)




        f_plus = np.asarray(layer.forward(x), dtype=np.float64)




        base[idx] = old - eps




        layer.forward(x)




        f_minus = np.asarray(layer.forward(x), dtype=np.float64)




        base[idx] = old




        num_grad[idx] = np.sum(dy * (f_plus - f_minus)) / (2 * eps)




    return num_grad














def _check_input_grad(layer, x, dy, eps=1e-5, label=""):
    """Analytical input gradient vs central-difference numerical gradient."""
    layer.forward(x)
    backward_out = layer.backward(dy)
    # Conv2D and Dense return a 3-tuple; ReLU/MaxPool/Flatten return a single
    # array (the input gradient) so it can be compared directly.
    an = backward_out[0] if isinstance(backward_out, tuple) else backward_out
    a = _num_grad_input(layer, x, dy, eps=eps)
    np.testing.assert_allclose(an, a, atol=1e-2, rtol=1e-2,
                                err_msg=f"{label or type(layer).__name__} input gradient")


def _check_weight_grad(layer, x, dy, param_name: str, eps=1e-5, label=""):




    """Weight gradient vs central-difference gradient of the scalar loss."""




    layer.forward(x)




    backward_out = layer.backward(dy)




    # Conv2D and Dense return a 3-tuple; only their weight gradient is checked.




    an = backward_out[1] if isinstance(backward_out, tuple) else backward_out




    ac = _num_grad_weight(layer, param_name, dy=dy, x=x, eps=eps)




    np.testing.assert_allclose(an, ac, atol=1e-2, rtol=1e-2,




                                err_msg=f"{label or type(layer).__name__} {param_name} gradient")














def test_conv2d_input_gradient(x_conv2d_small):




    layer = make_conv2d()




    layer.forward(x_conv2d_small)




    dy = np.array([[[[1.0]]]], dtype=np.float64)




    _check_input_grad(layer, x_conv2d_small, dy, eps=1e-5,




                      label="Conv2D input")














def test_conv2d_weight_gradient(x_conv2d_small):




    layer = make_conv2d()




    layer.forward(x_conv2d_small)




    dy = np.array([[[[1.0]]]], dtype=np.float64)




    _check_weight_grad(layer, x_conv2d_small, dy, "W",




                       label="Conv2D weight")














def test_conv2d_bias_gradient():




    x = np.array([[[[1.0], [2.0]], [[3.0], [4.0]]]], dtype=np.float64)




    layer = make_conv2d()




    layer.forward(x)




    dy = np.array([[[[1.0]]]], dtype=np.float64)




    an = layer.backward(dy)[2]




    base = np.copy(layer.b)




    num = np.zeros_like(base, dtype=np.float64)




    eps = 1e-6




    x0 = layer._cache["x"]




    for i in range(base.size):




        old = base[i]




        base[i] = old + eps




        layer.b = base




        layer._cache["x"] = x0.copy()




        f_plus = np.asarray(layer.forward(layer._cache["x"]), dtype=np.float64).ravel()




        base[i] = old - eps




        layer.b = base




        layer._cache["x"] = x0.copy()




        f_minus = np.asarray(layer.forward(layer._cache["x"]), dtype=np.float64).ravel()




        base[i] = old




        num[i] = np.sum(f_plus - f_minus) / (2 * eps)




    np.testing.assert_allclose(an, num, atol=1e-2, rtol=1e-2,




                                err_msg="Conv2D bias gradient")














def test_relu_input_gradient_outside_zero():
    dy = np.array([[1.0, 1.0, 1.0, 1.0]], dtype=np.float64)




    x = np.array([[-2.0, -0.5, 1.5, 3.0]], dtype=np.float64)




    layer = ReLU()




    layer._cache["x"] = x




    dy = np.array([[1.0, 1.0, 1.0, 1.0]], dtype=np.float64)




    _check_input_grad(layer, x, dy, eps=1e-5, label="ReLU")




    # x > 0 only at 1.5 and 3.0 (inputs 0 and 1 are non-positive)




    layer.forward(x)




    dx = layer.backward(dy)




    assert np.all(dx[0, 0:2] == 0.0)




    assert np.all(dx[0, 2:4] == 1.0)














def test_maxpool2d_input_gradient():




    x = np.array([[[[1.0], [2.0], [3.0], [4.0]],




                   [[5.0], [6.0], [7.0], [8.0]],




                   [[9.0], [10.0], [11.0], [12.0]],




                   [[13.0], [14.0], [15.0], [16.0]]]], dtype=np.float64)




    layer = MaxPool2D(pool_size=2, stride=2)




    layer.forward(x)




    dy = np.array([[[[1.0], [2.0]], [[3.0], [4.0]]]], dtype=np.float64)




    _check_input_grad(layer, x, dy, eps=1e-5, label="MaxPool2D")














def test_dense_input_gradient(x_dense_small):




    layer = make_dense()




    layer.W[1:, :] = 0.0




    layer.W[0, 1] = 0.0




    layer.forward(x_dense_small)




    dy = np.array([[1.0, 2.0]], dtype=np.float64)




    _check_input_grad(layer, x_dense_small, dy, eps=1e-5, label="Dense")














def test_dense_weight_gradient(x_dense_small):




    layer = make_dense()




    layer.W[1:, :] = 0.0




    layer.W[0, 1] = 0.0




    layer.forward(x_dense_small)




    dy = np.array([[1.0, 2.0]], dtype=np.float64)




    _check_weight_grad(layer, x_dense_small, dy, "W",




                       label="Dense")














def test_dense_bias_gradient(x_dense_small):
    layer = make_dense()
    layer.W[1:, :] = 0.0
    layer.W[0, 1] = 0.0
    layer.forward(x_dense_small)
    dy = np.array([[1.0, 2.0]], dtype=np.float64)
    an = layer.backward(dy)[2]
    base = np.copy(layer.b)
    num = np.zeros_like(base, dtype=np.float64)
    eps = 1e-6
    x0 = layer._cache["x"]
    for i in range(base.size):
        old = base[i]
        base[i] = old + eps
        layer.b = base
        layer._cache["x"] = x0.copy()
        f_plus = np.asarray(layer.forward(layer._cache["x"]), dtype=np.float64).ravel()
        base[i] = old - eps
        layer.b = base
        layer._cache["x"] = x0.copy()
        f_minus = np.asarray(layer.forward(layer._cache["x"]), dtype=np.float64).ravel()
        base[i] = old
        num[i] = np.sum(dy * (f_plus - f_minus)) / (2 * eps)
    np.testing.assert_allclose(an, num, atol=1e-2, rtol=1e-2,
                                err_msg="Dense bias gradient")















def test_stable_softmax_vector_jacobian_product():
    """StableSoftmax.backward is the general softmax vector-Jacobian product.

    For an arbitrary upstream gradient ``dP``, the layer returns
    ``dZ = P * (dP - sum(dP * P, axis=-1, keepdims=True))``.  It is not the
    fused softmax+cross-entropy shortcut.  The fused logits gradient
    ``(P - Y) / B`` is computed by :class:`CategoricalCrossEntropy.backward`
    and by :class:`ScratchCNN.backward`.
    """
    layer = StableSoftmax()
    z = np.array([[2.0, 1.0, 0.0],
                  [0.5, 1.0, 3.0]], dtype=np.float64)
    layer.forward(z)
    p = layer._cache["p"]
    rng = np.random.default_rng(7)
    dP = rng.standard_normal(z.shape).astype(np.float64)

    an = layer.backward(dP)

    # The mathematical VJP.
    expected_vjp = p * (dP - np.sum(dP * p, axis=-1, keepdims=True))
    np.testing.assert_allclose(an, expected_vjp, atol=1e-12, rtol=1e-12,
                                err_msg="stable softmax VJP matches P*(dP - sum(dP*P))")

    # The analytical gradient is the numerical gradient of the scalar loss
    # ``L = sum(dy * f)`` w.r.t. the logits z.
    base = z.copy()
    num_dz = np.zeros_like(base, dtype=np.float64)
    for idx_ep in np.ndindex(base.shape):
        old = base[idx_ep]
        base[idx_ep] = old + 1e-5
        f_plus = np.asarray(layer.forward(base), dtype=np.float64)
        base[idx_ep] = old - 1e-5
        f_minus = np.asarray(layer.forward(base), dtype=np.float64)
        num_dz[idx_ep] = np.sum(dP * (f_plus - f_minus)) / (2 * 1e-5)
        base[idx_ep] = old
    np.testing.assert_allclose(an, num_dz, atol=1e-2, rtol=1e-2,
                                err_msg="stable softmax VJP vs numerical gradient")


def test_categorical_crossentropy_logits_gradient():
    """CategoricalCrossEntropy.backward() is the fused logits gradient (P-Y)/B.

    The mean cross-entropy loss is L = -(1/B) sum_b Y_b . log P_b.  Chaining
    the stable softmax Jacobian with the one-hot cross-entropy collapses to
    dZ = (P - Y) / B (Section 14 of the mathematical foundations).  This is a
    single application of the batch averaging at the loss level; the softmax
    layer's backward must not be applied again on top.
    """
    loss = CategoricalCrossEntropy()
    y = np.array([[1.0, 0.0, 0.0],
                  [0.0, 1.0, 0.0]], dtype=np.float64)
    z = np.array([[0.9, 0.1, -0.3],
                  [0.2, 1.5, -0.7]], dtype=np.float64)

    # Recompute the probability path to obtain P from Z.
    z_shift = z - np.max(z, axis=-1, keepdims=True)
    e = np.exp(z_shift)
    p_from_z = e / np.sum(e, axis=-1, keepdims=True)

    # Numerical gradient of the mean loss w.r.t. the logits.
    eps = 1e-6
    num_dz = np.zeros_like(z, dtype=np.float64)
    for idx_ep in np.ndindex(z.shape):
        zp = z.copy(); zp[idx_ep] += eps
        zm = z.copy(); zm[idx_ep] -= eps
        zp_s = zp - np.max(zp, axis=-1, keepdims=True)
        zm_s = zm - np.max(zm, axis=-1, keepdims=True)
        ep = np.exp(zp_s); ep /= ep.sum(-1, keepdims=True)
        em = np.exp(zm_s); em /= em.sum(-1, keepdims=True)
        num_dz[idx_ep] = (loss.forward(ep, y) - loss.forward(em, y)) / (2 * eps)

    # Feed the recomputed probabilities into the loss so backward() sees the
    # right P (the numerical loop above overwrites the cache).
    loss.forward(p_from_z, y)
    an = loss.backward()
    np.testing.assert_allclose(an, num_dz, atol=1e-2, rtol=1e-2,
                                err_msg="categorical cross-entropy logits gradient")
    np.testing.assert_allclose(an, (p_from_z - y) / z.shape[0], atol=1e-12,
                                err_msg="cross-entropy backward == (P - Y) / B")


def test_scratchcnn_fused_path():
    """ScratchCNN.backward(dy=None) matches the explicit fused (P-Y)/B gradient.

    The frozen architecture's backward must not route the logits gradient
    through StableSoftmax.backward (which would apply the softmax derivative
    a second time).  Passing dy=None must compute the same fused gradient as
    dy=(P-Y)/B.
    """
    class MiniScratch(ScratchCNN):
        def __init__(self):
            super().__init__([
                {"type": "conv2d", "kernel_size": 3, "channels": 1, "filters": 1,
                 "use_bias": True},
                {"type": "relu"},
                {"type": "maxpool2d", "pool_size": 2, "stride": 2},
                {"type": "flatten"},
                {"type": "dense", "inputs": 4, "units": 2, "use_bias": True},
                {"type": "softmax"},
            ], seed=3)

    x = np.zeros((1, 6, 6, 1), dtype=np.float64)
    for i in range(6):
        for j in range(6):
            x[0, i, j, 0] = float(i + j)
    y = np.zeros((1, 2), dtype=np.float64)
    y[0, 1] = 1.0

    model = MiniScratch()
    probs, loss_val = model.forward(x, y)

    # Backward with dy=None (computes (P-Y)/B internally).
    g_none = model.backward(None, y)

    # Backward with the explicit fused dy.
    g_explicit = model.backward((probs - y), y)

    # Both must agree with a manual backprop of the fused logits gradient.
    dy = (probs - y) / probs.shape[0]
    _, dW2_manual, db2_manual = model.dense.backward(dy)
    dx = model.dense.backward(dy)[0]
    dx = model.flatten.backward(dx)
    dx = model.pool.backward(dx)
    dx = model.relu.backward(dx)
    dx, dW1_manual, db1_manual = model.conv.backward(dx)

    np.testing.assert_allclose(g_none["W2"], dW2_manual, atol=1e-12,
                                err_msg="ScratchCNN W2 matches manual fused backprop")
    np.testing.assert_allclose(g_none["b2"], db2_manual, atol=1e-12,
                                err_msg="ScratchCNN b2 matches manual fused backprop")
    np.testing.assert_allclose(g_none["x"], dx, atol=1e-12,
                                err_msg="ScratchCNN x matches manual fused backprop")
    np.testing.assert_allclose(g_none["x"], g_explicit["x"], atol=1e-12,
                                err_msg="dy=None agrees with explicit fused dy")
    np.testing.assert_allclose(g_none["W1"], dW1_manual, atol=1e-12,
                                err_msg="ScratchCNN W1 matches manual fused backprop")
    np.testing.assert_allclose(g_none["b1"], db1_manual, atol=1e-12,
                                err_msg="ScratchCNN b1 matches manual fused backprop")




# ---------------------------------------------------------------------------




# Deliberate tie and overlapping-window checks




# ---------------------------------------------------------------------------














def test_maxpool2d_tie_uses_first_maximum():




    # A 2x2 window where every value is identical: the first maximum in




    # row-major order (local (0,0)) is the winner.




    x = np.array([[[[5.0], [5.0]], [[5.0], [5.0]]]], dtype=np.float64)




    layer = MaxPool2D(pool_size=2, stride=2)




    layer.forward(x)




    # Upstream gradient for the single pooled output.




    dy = np.array([[[[2.0]]]], dtype=np.float64)




    dx = layer.backward(dy)




    # The single 2x2 window's max is at local (0, 0) -> input (0, 0).




    assert dx[0, 0, 0, 0] == 2.0




    assert np.count_nonzero(dx) == 1














def test_conv2d_overlapping_windows_accumulate():
    """Overlapping windows accumulate gradient additively.

    Kernel (2, 2) on a (4, 4) input yields a (3, 3) output.  With
    ``dy = ones((1, 3, 3, 1))`` each output window contributes its full x
    patch, so the kernel gradient at ``(u, v)`` is the sum of the x values in
    the patch shifted by ``(u, v)`` across all nine windows:

        dW[u, v] = sum_{i,j} x[i+u, j+v]

    With ``x[b, i, j] = i + j`` this gives
    ``dW[u, v] = 9*(u + v) + 18``.
    """
    x = np.zeros((1, 4, 4, 1), dtype=np.float64)
    for i in range(4):
        for j in range(4):
            x[0, i, j, 0] = float(i + j)
    layer = make_conv2d()
    layer.W[:, :, 0, 0] = 1.0
    dy = np.ones((1, 3, 3, 1), dtype=np.float64)
    layer.forward(x)
    _, dw, _ = layer.backward(dy)
    # dW[u, v] = 9*(u+v) + 18
    assert dw[0, 0, 0, 0] == pytest.approx(18.0)
    assert dw[0, 1, 0, 0] == pytest.approx(27.0)
    assert dw[1, 0, 0, 0] == pytest.approx(27.0)
    assert dw[1, 1, 0, 0] == pytest.approx(36.0)


def test_maxpool2d_overlapping_stride_forward():
    """Stride smaller than the pool size pools overlapping windows.

    ``pool_size=2, stride=1`` on a 4x4 input yields a 3x3 output; window
    ``(i, j)`` covers input rows ``i..i+1`` and columns ``j..j+1``.
    """
    x = np.arange(16, dtype=np.float64).reshape(1, 4, 4, 1)
    layer = MaxPool2D(pool_size=2, stride=1)
    y = layer.forward(x)
    assert y.shape == (1, 3, 3, 1)
    expected = np.array([[[[5.0], [6.0], [7.0]],
                          [[9.0], [10.0], [11.0]],
                          [[13.0], [14.0], [15.0]]]], dtype=np.float64)
    np.testing.assert_allclose(y, expected)


def test_maxpool2d_overlapping_stride_backward_accumulates():
    """Routing and accumulation of overlapping-window gradients.

    With ``pool_size=2, stride=1`` windows overlap: each interior cell is
    covered by up to four windows, but a gradient is routed only to each
    window's argmax, and cells that win several windows must *sum* the
    contributions instead of overwriting them.  The first fixture checks
    the routing map and gradient conservation; the second pins a shared
    argmax (two windows with the same maximum cell) and the routing of a
    window whose region also contains a cell won by a neighbour.
    """
    x = np.arange(16, dtype=np.float64).reshape(1, 4, 4, 1)
    layer = MaxPool2D(pool_size=2, stride=1)
    layer.forward(x)
    dy = np.ones((1, 3, 3, 1), dtype=np.float64)
    dx = layer.backward(dy)
    # Every window has a distinct maximum here, so each winning cell (the
    # argmax of exactly one window) receives 1.0 and all others stay 0.
    expected_dx = np.array([[[[0.0], [0.0], [0.0], [0.0]],
                             [[0.0], [1.0], [1.0], [1.0]],
                             [[0.0], [1.0], [1.0], [1.0]],
                             [[0.0], [1.0], [1.0], [1.0]]]], dtype=np.float64)
    np.testing.assert_allclose(dx, expected_dx)
    # Pooling routes every upstream gradient exactly once: totals match.
    assert dx.sum() == pytest.approx(dy.sum())

    # Routing under overlap: the window at (0, 1) must route to its own
    # maximum (0, 2), not to (0, 1) whose flag came from the window at (0, 0).
    x2 = np.array([[[[1.0], [5.0], [10.0]],
                    [[2.0], [3.0], [0.0]],
                    [[0.0], [0.0], [0.0]]]], dtype=np.float64)
    layer2 = MaxPool2D(pool_size=2, stride=1)
    y2 = layer2.forward(x2)
    np.testing.assert_allclose(
        y2, np.array([[[[5.0], [10.0]], [[3.0], [3.0]]]]))
    dx2 = layer2.backward(np.ones((1, 2, 2, 1), dtype=np.float64))
    expected_dx2 = np.array([[[[0.0], [1.0], [1.0]],
                              [[0.0], [2.0], [0.0]],
                              [[0.0], [0.0], [0.0]]]], dtype=np.float64)
    np.testing.assert_allclose(dx2, expected_dx2)
