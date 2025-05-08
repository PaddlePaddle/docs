.. _api_guide_activations_en:

###################
Activation Function
###################

The activation function incorporates non-linearity properties into the neural network.

PaddlePaddle supports most of the activation functions, including:

* :ref:`api_paddle_nn_functional_elu`
* :ref:`api_paddle_exp`
* :ref:`api_paddle_nn_functional_hardsigmoid`
* :ref:`api_paddle_nn_functional_hardshrink`
* :ref:`api_paddle_nn_functional_leaky_relu`
* :ref:`api_paddle_nn_functional_log_sigmoid`
* :ref:`api_paddle_nn_functional_maxout`
* :ref:`api_paddle_pow`
* :ref:`api_paddle_nn_functional_prelu`
* :ref:`api_paddle_nn_functional_relu`
* :ref:`api_paddle_nn_functional_relu6`
* :ref:`api_paddle_tensor_sigmoid`
* :ref:`api_paddle_nn_functional_softplus`
* :ref:`api_paddle_nn_functional_softshrink`
* :ref:`api_paddle_nn_functional_softsign`
* :ref:`api_paddle_stanh`
* :ref:`api_paddle_nn_functional_swish`
* :ref:`api_paddle_tanh`
* :ref:`api_paddle_nn_functional_thresholded_relu`
* :ref:`api_paddle_nn_functional_tanhshrink`


**The way to apply activation functions in PaddlePaddle is as follows:**

PaddlePaddle provides a dedicated interface for each activation function, allowing users to explicitly invoke them as needed. Below is an example of applying the ReLU activation function after a convolution operation:

.. code-block:: python

    conv2d = paddle.nn.functional.conv2d(x, weight, stride=1, padding=1)  # Convolution operation
    relu1 = paddle.nn.functional.relu(conv2d)  # Applying the ReLU activation function
