.. _cn_api_nn_Tanh:

Tanh
-------------------------------
.. py:class:: paddle.nn.Tanh(name=None)

Tanh激活层

.. math::
    Tanh(x) = \frac{e^{x} - e^{-x}}{e^{x} + e^{-x}}


参数
::::::::::
    - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。

形状
::::::::::

    - input：任意形状的Tensor。
    - output：和input具有相同形状的Tensor。

代码示例
::::::::::

COPY-FROM: paddle.nn.Tanh