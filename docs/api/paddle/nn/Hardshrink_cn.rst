.. _cn_api_paddle_nn_Hardshrink:

Hardshrink
-------------------------------
.. py:class:: paddle.nn.Hardshrink(threshold=0.5, name=None)

Hardshrink 激活层

.. math::

    Hardshrink(x)=
        \left\{
        \begin{aligned}
        &x, & & if \ x > threshold \\
        &x, & & if \ x < -threshold \\
        &0, & & if \ others
        \end{aligned}
        \right.

其中，:math:`x` 为输入的 Tensor。

参数
::::::::::
    - **threshold** (float，可选) - Hardshrink 激活计算公式中的 threshold 值。默认值为 0.5。
    - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。

形状
::::::::::
    - input：任意形状的 Tensor。
    - output：和 input 具有相同形状的 Tensor。

代码示例
::::::::::

COPY-FROM: paddle.nn.Hardshrink
