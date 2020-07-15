import pytest
import tensorflow as tf
import numpy as np

from loss.differentiable_argmax_approx import differentiable_argmax_approx


@pytest.fixture
def session():
    session = tf.Session()
    yield session
    session.close()


def test_argmax_approx_single_max(session):
    x = tf.placeholder(dtype=tf.float64, shape=(None,))
    beta = 100
    y = differentiable_argmax_approx(x, beta)
    data = np.array([1, 2, 3, 10, 4, 5])
    assert data.argmax() == int(session.run(y, feed_dict={x: data / np.linalg.norm(data)}))


def test_argmax_approx_multiple_max(session):
    x = tf.placeholder(dtype=tf.float64, shape=(None,))
    beta = 100
    y = differentiable_argmax_approx(x, beta)
    data = np.array([1, 2, 10, 3, 10])
    assert int(session.run(y, feed_dict={x: data / np.linalg.norm(data)})) == 3
