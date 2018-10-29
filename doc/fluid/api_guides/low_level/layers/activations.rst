.. _api_guide_activations:

####
激活函数
#### 

激活函数将非线性的特性引入到神经网络当中。

PaddlePaddle Fluid 对大部分的激活函数进行了支持，其中有:        

:code:`relu`, :code:`tanh`, :code:`sigmoid`, :code:`elu`, :code:`relu6`, :code:`pow`, :code:`stanh`, :code:`hard_sigmoid`, :code:`swish`, :code:`prelu`, :code:`brelu`, :code:`leaky_relu`, :code:`soft_relu`, :code:`thresholded_relu`, :code:`maxout`, :code:`logsigmoid`, :code:`hard_shrink`, :code:`softsign`, :code:`softplus`, :code:`tanh_shrink`, :code:`softshrink`, :code:`exp`, :code:`identity`。
 

Fluid提供了两种使用激活函数的方式：
==============

- 如果一个层的接口提供了 :code:`act` 变量（默认值为None），我们可以通过该变量指定该层的激活函数类型。该方式支持常见的激活函数，:code:`relu`, :code:`tanh`, :code:`sigmoid`, :code:`identity`。

.. code-block:: python

	conv2d = fluid.layers.conv2d(input=data, num_filters=2, filter_size=3, act="relu")


- Fluid为每个Activation提供了接口，我们可以显式的对它们进行调用。

.. code-block:: python

	conv2d = fluid.layers.conv2d(input=data, num_filters=2, filter_size=3)
	relu1 = fluid.layers.relu(conv2d)
