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















def test_stable_softmax_crossentropy_logits_gradient():
    """The combined softmax+CE gradient w.r.t. the logits is (P - Y) / B.

    StableSoftmax.backward implements the combined softmax + cross-entropy
    backward: it is handed the gradient of the mean loss w.r.t. the logits,
    ``(P - Y) / B``, and returns it unchanged (the softmax and the loss are
    already one graph, so no second differentiation happens here).  Feeding
    that gradient through the layer must reproduce the numerical gradient of
    the loss w.r.t. ``z`` exactly.
    """
    layer = StableSoftmax()
    y = np.array([[1.0, 0.0, 0.0],
                  [0.0, 1.0, 0.0]], dtype=np.float64)
    z = np.array([[2.0, 1.0, 0.0],
                  [0.5, 1.0, 3.0]], dtype=np.float64)
    loss = CategoricalCrossEntropy()
    p = layer.forward(z)
    loss.forward(p, y)
    # Gradient of the mean loss w.r.t. the logits: (P - Y) / B.
    dy = (p - y) / p.shape[0]
    dz_analytical = layer.backward(dy)

    base = z.copy()
    num_dz = np.zeros_like(base, dtype=np.float64)
    for idx in np.ndindex(base.shape):
        old = base[idx]
        base[idx] = old + 1e-5
        p_plus = layer.forward(base)
        lv_plus = CategoricalCrossEntropy().forward(p_plus, y)
        base[idx] = old - 1e-5
        p_minus = layer.forward(base)
        lv_minus = CategoricalCrossEntropy().forward(p_minus, y)
        base[idx] = old
        num_dz[idx] = (lv_plus - lv_minus) / (2e-5)
    np.testing.assert_allclose(dz_analytical, num_dz, atol=1e-2, rtol=1e-2,
                                err_msg="Softmax+CE logits gradient")















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










