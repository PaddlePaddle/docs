.. _cn_api_paddle_nn_Sigmoid:

Sigmoid
-------------------------------

.. py:class:: paddle.nn.Sigmoid(name=None)

用于创建一个 ``Sigmoid`` 的可调用类。这个类可以计算输入 :attr:`x` 经过激活函数 ``sigmoid`` 之后的值。

    .. math::

        sigmoid(x) = \frac{1}{1 + \mathrm{e}^{-x}}

参数
::::::::

  - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。

形状
::::::::

  - **x** (Tensor) - N-D Tensor，支持的数据类型是 float16、float32 和 float64。

返回
::::::::

返回计算 ``Sigmoid`` 的可调用对象。


代码示例
::::::::

COPY-FROM: paddle.nn.Sigmoid
