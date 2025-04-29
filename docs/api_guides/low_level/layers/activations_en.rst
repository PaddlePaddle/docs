.. _api_guide_activations_en:

###################
Activation Function
###################

The activation function incorporates non-linearity properties into the neural network.

PaddlePaddle supports most of the activation functions, including:

- :ref:`api_exp`
- :ref:`api_pow`
- :ref:`api_stanh`
- :ref:`api_nn_functional_elu`
- :ref:`api_nn_functional_hard_sigmoid`
- :ref:`api_nn_functional_hard_shrink`
- :ref:`api_nn_functional_leaky_relu`
- :ref:`api_nn_functional_logsigmoid`
- :ref:`api_nn_functional_maxout`
- :ref:`api_nn_functional_prelu`
- :ref:`api_static_nn_prelu`
- :ref:`api_nn_functional_relu`
- :ref:`api_nn_functional_relu6`
- :ref:`api_nn_functional_sigmoid`
- :ref:`api_nn_functional_softplus`
- :ref:`api_nn_functional_softshrink`
- :ref:`api_nn_functional_softsign`
- :ref:`api_nn_functional_swish`
- :ref:`api_nn_functional_thresholded_relu`
- :ref:`api_nn_functional_tanh`
- :ref:`api_nn_functional_tanh_shrink`


**PaddlePaddle provides two ways to use the activation function:**

- If a layer interface provides :code:`act` variables (default None), we can specify the type of layer activation function through this parameter. This mode supports common activation functions :code:`relu`, :code:`tanh`, :code:`sigmoid`, :code:`identity`.

.. code-block:: python

    conv2d = nn.functional.conv2d(input=data, num_filters=2, filter_size=3, act="relu")


- PaddlePaddle provides an interface for each Activation, and we can explicitly call it.

.. code-block:: python

    conv2d = nn.functional.conv2d(input=data, num_filters=2, filter_size=3)
    relu1 = nn.functional.relu(conv2d)
