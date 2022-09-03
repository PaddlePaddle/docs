.. _cn_api_nn_Swish:

Swish
-------------------------------
.. py:class:: paddle.nn.Swish(name=None)

Swish 激活层

.. math::

    Swish(x) = \frac{x}{1 + e^{-x}}

其中，:math:`x` 为输入的 Tensor

参数
::::::::::
    - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。

形状：
::::::::::
    - input：任意形状的 Tensor。
    - output：和 input 具有相同形状的 Tensor。

代码示例
:::::::::

COPY-FROM: paddle.nn.Swish
