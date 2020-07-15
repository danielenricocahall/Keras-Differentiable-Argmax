# Keras-Differentiable-Argmax #

To work around the fact that `K.argmax`/`tf.argmax` is not differentiable, here is an approximation to the `argmax` function which is tunable by parameter beta. The implementation is based on a StackOverflow discussion (https://stackoverflow.com/questions/46926809/getting-around-tf-argmax-which-is-not-differentiable), and I figured it would be worth having cleaned up/usable in one place. To show an example of how it worked, I have provided two test cases. If you have any questions or suggestions, please feel free to let me know!

## Running Tests ##

To run the tests, install the virtual environment with `pipenv install .`, navigate into the environment (`pipenv shell`) or configure the environment in your IDE, then run `pytest .` in the project directory. Both tests should execute and pass.

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


## Limitations ##

If the maximum value is not unique along the axis that we're applying the argmax function to, the result will be an average between the two indices, as shown in `test_argmax_approx_multiple_max`. The regular `argmax` has a different limitation, as it typically returns the first index of the maximum argument. However, that is arguably more correct then averaging the indices.
