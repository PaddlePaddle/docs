.. _api_guide_activations:

####
Activation Function
####

The activation function introduces nonlinear properties into the neural network.

PaddlePaddle Fluid supports most of the activation functions, including:

:ref:`cn_api_fluid_layers_relu`,
:ref:`cn_api_fluid_layers_tanh`,
:ref:`cn_api_fluid_layers_sigmoid`,
:ref:`cn_api_fluid_layers_elu`,
:ref: `cn_api_fluid_layers_relu6`,
:ref:`cn_api_fluid_layers_pow`,
:ref:`cn_api_fluid_layers_stanh`,
:ref:`cn_api_fluid_layers_hard_sigmoid`,
:ref:`cn_api_fluid_layers_swish`,
:ref:`cn_api_fluid_layers_prelu`,
:ref:`cn_api_fluid_layers_brelu`,
:ref:`cn_api_fluid_layers_leaky_relu`,
:ref:`cn_api_fluid_layers_soft_relu`,
:ref:`cn_api_fluid_layers_thresholded_relu`,
:ref :`cn_api_fluid_layers_maxout`,
:ref:`cn_api_fluid_layers_logsigmoid`,
:ref:`cn_api_fluid_layers_hard_shrink`,
:ref:`cn_api_fluid_layers_softsign`,
:ref:`cn_api_fluid_layers_softplus`,
:ref:`cn_api_fluid_layers_tanh_shrink`,
:ref:`cn_api_fluid_layers_softshrink`,
:ref:`cn_api_fluid_layers_exp`.


**Fluid provides two ways to use the activation function:**

- If a layer interface provides :code:`act` variables (default None), we can specify the type of layer activation function by the variable. This mode supports common activation functions :code:`relu`, :code:`tanh`, :code:`sigmoid`, :code:`identity`.

.. code-block:: python

	conv2d = fluid.layers.conv2d(input=data, num_filters=2, filter_size=3, act="relu")


- Fluid provides an interface for each Activation, and we can explicitly call them.

.. code-block:: python

	conv2d = fluid.layers.conv2d(input=data, num_filters=2, filter_size=3)
	relu1 = fluid.layers.relu(conv2d)
