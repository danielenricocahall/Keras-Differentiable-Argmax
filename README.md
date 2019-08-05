# Keras-Differentiable-Argmax #

To work around the fact that K.argmax/tf.argmax is not differentiable, here is an approximation to the argmax function which is tunable by parameter beta. The implementation is based on a StackOverflow discussion (https://stackoverflow.com/questions/46926809/getting-around-tf-argmax-which-is-not-differentiable), and I figured it would be worth having cleaned up/usable in one place. To show an example of how it worked, I have also provided a simple demo script. If you have any questions or suggestions, please feel free to let me know!

## Example Output ##
| Argmax | Argmax Approx. |
| --- | --- |
|7| 6.999999996348694|
|8| 7.748457525423447|
|1| 1.0000000572854653|
|8| 7.999999525194022|
|8| 7.999992630730205|
|5| 5.000000000000002|
|5| 5.12267064474127|
|3| 2.9999998129252115|
|7| 6.599979662770566|
|3| 3.0000489993240986|
