import tensorflow as tf


def differentiable_argmax_approx(x, beta=1e2):
    """
    Approximation of argmax - translates to y = sum( i * exp(beta * x[i]) ) / sum( exp(beta * x[i]) )
    :param x:
    :param beta:
    :return:
    """
    y = tf.reduce_sum(tf.cumsum(tf.ones_like(x)) * tf.exp(beta * x) / tf.reduce_sum(tf.exp(beta * x))) - 1
    return y
