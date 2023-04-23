.. _api_guide_activations_en:

###################
Activation Function
###################

The activation function incorporates non-linearity properties into the neural network.

PaddlePaddle Fluid supports most of the activation functions, including:

:ref:`api_fluid_layers_relu`,
:ref:`api_fluid_layers_tanh`,
:ref:`api_fluid_layers_sigmoid`,
:ref:`api_fluid_layers_elu`,
:ref:`api_fluid_layers_relu6`,
:ref:`api_fluid_layers_pow`,
:ref:`api_fluid_layers_stanh`,
:ref:`api_fluid_layers_hard_sigmoid`,
:ref:`api_fluid_layers_swish`,
:ref:`api_fluid_layers_prelu`,
:ref:`api_fluid_layers_brelu`,
:ref:`api_fluid_layers_leaky_relu`,
:ref:`api_fluid_layers_soft_relu`,
:ref:`api_fluid_layers_thresholded_relu`,
:ref:`api_fluid_layers_maxout`,
:ref:`api_fluid_layers_logsigmoid`,
:ref:`api_fluid_layers_hard_shrink`,
:ref:`api_fluid_layers_softsign`,
:ref:`api_fluid_layers_softplus`,
:ref:`api_fluid_layers_tanh_shrink`,
:ref:`api_fluid_layers_softshrink`,
:ref:`api_fluid_layers_exp`.


**Fluid provides two ways to use the activation function:**

- If a layer interface provides :code:`act` variables (default None), we can specify the type of layer activation function through this parameter. This mode supports common activation functions :code:`relu`, :code:`tanh`, :code:`sigmoid`, :code:`identity`.

.. code-block:: python

    conv2d = fluid.layers.conv2d(input=data, num_filters=2, filter_size=3, act="relu")


- Fluid provides an interface for each Activation, and we can explicitly call it.

.. code-block:: python

    conv2d = fluid.layers.conv2d(input=data, num_filters=2, filter_size=3)
    relu1 = fluid.layers.relu(conv2d)
