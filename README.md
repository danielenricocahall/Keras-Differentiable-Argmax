# Keras-Differentiable-Argmax #

To work around the fact that K.argmax/tf.argmax is not differentiable, here is an approximation to the argmax function which is tunable by parameter beta. The implementation is based on a StackOverflow discussion (https://stackoverflow.com/questions/46926809/getting-around-tf-argmax-which-is-not-differentiable), and I figured it would be worth having cleaned up/usable in one place. To show an example of how it worked, I have also provided a simple demo script. If you have any questions or suggestions, please feel free to let me know!
