.. _cn_api_paddle_nn_Tanhshrink:

Tanhshrink
-------------------------------
.. py:class:: paddle.nn.Tanhshrink(name=None)

Tanhshrink 激活层

.. math::

    Tanhshrink(x) = x - tanh(x)

其中，:math:`x` 为输入的 Tensor。

参数
::::::::::
    - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。

形状
::::::::::
    - input：任意形状的 Tensor。
    - output：和 input 具有相同形状的 Tensor。

代码示例
:::::::::

COPY-FROM: paddle.nn.Tanhshrink
