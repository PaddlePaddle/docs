.. _api_guide_activations:

###################
激活函数
###################

激活函数将非线性的特性引入到神经网络当中。

PaddlePaddle 对大部分的激活函数进行了支持，其中有:

- :ref:`cn_api_exp`
- :ref:`cn_api_pow`
- :ref:`cn_api_stanh`
- :ref:`cn_api_nn_functional_elu`
- :ref:`cn_api_nn_functional_hard_sigmoid`
- :ref:`cn_api_nn_functional_hard_shrink`
- :ref:`cn_api_nn_functional_leaky_relu`
- :ref:`cn_api_nn_functional_logsigmoid`
- :ref:`cn_api_nn_functional_maxout`
- :ref:`cn_api_nn_functional_prelu`
- :ref:`cn_api_static_nn_prelu`
- :ref:`cn_api_nn_functional_relu`
- :ref:`cn_api_nn_functional_relu6`
- :ref:`cn_api_nn_functional_sigmoid`
- :ref:`cn_api_nn_functional_softplus`
- :ref:`cn_api_nn_functional_softshrink`
- :ref:`cn_api_nn_functional_softsign`
- :ref:`cn_api_nn_functional_swish`
- :ref:`cn_api_nn_functional_thresholded_relu`
- :ref:`cn_api_nn_functional_tanh`
- :ref:`cn_api_nn_functional_tanh_shrink`


**PaddlePaddle 提供了两种使用激活函数的方式：**

- 如果一个层的接口提供了 :code:`act` 变量（默认值为 None），我们可以通过该变量指定该层的激活函数类型。该方式支持常见的激活函数: :code:`relu`, :code:`tanh`, :code:`sigmoid`, :code:`identity`。

.. code-block:: python

    conv2d = nn.functional.conv2d(input=data, num_filters=2, filter_size=3, act="relu")


- PaddlePaddle 为每个 Activation 提供了接口，我们可以显式的对它们进行调用。


.. code-block:: python

    conv2d = nn.functional.conv2d(input=data, num_filters=2, filter_size=3)
    relu1 = nn.functional.relu(conv2d)
