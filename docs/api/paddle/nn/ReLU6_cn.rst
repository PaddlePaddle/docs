.. _cn_api_nn_ReLU6:

ReLU6
-------------------------------
.. py:class:: paddle.nn.ReLU6(name=None)

ReLU6 激活层

.. math::

    ReLU6(x) = min(max(0,x), 6)

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

COPY-FROM: paddle.nn.ReLU6
