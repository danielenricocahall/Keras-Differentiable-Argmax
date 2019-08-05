'''
Created on Aug 5, 2019

@author: DCahall
'''
import tensorflow as tf
import numpy as np
from DifferentiableArgmaxApproximation.DifferentiableArgmaxApproximation import DifferentiableArgmaxApproximation


def main():
    sess = tf.Session()
    x = tf.placeholder(dtype=tf.float64, shape=(None,))
    
    beta = 100
    y = DifferentiableArgmaxApproximation(x, beta)
    
    print("I can compute the gradient", tf.gradients(y, x))
    
    ## Compare the actual argmax to the approximation
    ## Should be fairly close (assuming there is one unique argmax)
    for _ in range(10):
        data = np.random.randn(10)
        print(data.argmax(), sess.run(y, feed_dict={x:data/np.linalg.norm(data)}))
        
if __name__=="__main__":
    main()
    exit()
    