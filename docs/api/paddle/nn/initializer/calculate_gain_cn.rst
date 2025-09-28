.. _cn_api_paddle_nn_initializer_calculate_gain:

calculate_gain
-------------------------------

.. py:function:: paddle.nn.initializer.calculate_gain(nonlinearity, param=None)

部分激活函数的推荐增益值（增益值可用于设置某些初始化 API，以调整初始化值）。

参数
:::::::::
    - **nonlinearity** (str) - 非线性激活函数的名称。如果输入一个线性的函数，例如：`linear/conv1d/conv2d/conv3d/conv1d_transpose/conv2d_transpose/conv3d_transpose`，则返回 1.0。
    - **param** (bool|int|float，可选) - 某些激活函数的参数，目前仅用于 ``leaky_relu`` 中的计算。默认为 ``None``，此时以 0.01 来参与 ``leaky_relu`` 的增益值计算。

返回
:::::::::
Python float 数，推荐的增益值。

代码示例
:::::::::

COPY-FROM: paddle.nn.initializer.calculate_gain
