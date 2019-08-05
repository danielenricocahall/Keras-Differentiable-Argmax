import tensorflow as tf

# Pseudo-math for the below
# y = sum( i * exp(beta * x[i]) ) / sum( exp(beta * x[i]) )
def DifferentiableArgmaxApproximation(x, beta = 1e2):
    y = tf.reduce_sum(tf.cumsum(tf.ones_like(x)) * tf.exp(beta * x) / tf.reduce_sum(tf.exp(beta * x))) - 1
    return y

