import numpy as np

from loss.differentiable_argmax_approx import differentiable_argmax_approx


def test_argmax_approx_single_max():
    beta = 10
    x = np.array([1, 2, 3, 10, 4, 5], dtype=np.float)
    y = differentiable_argmax_approx(x, beta)
    assert x.argmax() == y


def test_argmax_approx_multiple_max():
    beta = 10
    x = np.array([1, 2, 10, 3, 10], dtype=np.float)
    y = differentiable_argmax_approx(x, beta)
    assert y == 3
