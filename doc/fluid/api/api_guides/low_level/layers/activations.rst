.. _api_guide_activations:

####
激活函数
#### 

激活函数将非线性的特性引入到神经网络当中。

PaddlePaddle Fluid 对大部分的激活函数进行了支持，其中有:        

:ref:`api_fluid_layers_relu`, :ref:`api_fluid_layers_tanh`, :ref:`api_fluid_layers_sigmoid`, :ref:`api_fluid_layers_elu`, :ref:`api_fluid_layers_relu6`, :ref:`api_fluid_layers_pow`, :ref:`api_fluid_layers_stanh`, :ref:`api_fluid_layers_hard_sigmoid`, :ref:`api_fluid_layers_swish`, :ref:`api_fluid_layers_prelu`, :ref:`api_fluid_layers_brelu`, :ref:`api_fluid_layers_leaky_relu`, :ref:`api_fluid_layers_soft_relu`, :ref:`api_fluid_layers_thresholded_relu`, :ref:`api_fluid_layers_maxout`, :ref:`api_fluid_layers_logsigmoid`, :ref:`api_fluid_layers_hard_shrink`, :ref:`api_fluid_layers_softsign`, :ref:`api_fluid_layers_softplus`, :ref:`api_fluid_layers_tanh_shrink`, :ref:`api_fluid_layers_softshrink`, :ref:`api_fluid_layers_exp`。
 

**Fluid提供了两种使用激活函数的方式：**

- 如果一个层的接口提供了 :code:`act` 变量（默认值为None），我们可以通过该变量指定该层的激活函数类型。该方式支持常见的激活函数: :code:`relu`, :code:`tanh`, :code:`sigmoid`, :code:`identity`。

.. code-block:: python

	conv2d = fluid.layers.conv2d(input=data, num_filters=2, filter_size=3, act="relu")


- Fluid为每个Activation提供了接口，我们可以显式的对它们进行调用。

.. code-block:: python

	conv2d = fluid.layers.conv2d(input=data, num_filters=2, filter_size=3)
	relu1 = fluid.layers.relu(conv2d)
