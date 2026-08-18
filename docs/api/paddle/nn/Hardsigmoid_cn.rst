.. _cn_api_paddle_nn_Hardsigmoid:

Hardsigmoid
-------------------------------

.. py:class:: paddle.nn.Hardsigmoid(inplace=False, name=None)

Hardsigmoid 激活层，用于创建一个 ``Hardsigmoid`` 类的可调用对象。sigmoid 的分段线性逼近激活函数，速度比 sigmoid 快，详细解释参见 `Noisy Activation Functions <https://arxiv.org/abs/1603.00391>`_ 。

.. math::

    Hardsigmoid(x)=
        \left\{
        \begin{aligned}
        &0, & & \text{if } x \leq -3 \\
        &1, & & \text{if } x \geq 3 \\
        &x/6 + 1/2, & & \text{otherwise}
        \end{aligned}
        \right.

其中，:math:`x` 为输入的 Tensor。

参数
::::::::::

    - **inplace** (bool，可选) - 是否使用原地操作。默认值为 False。
    - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。

形状
::::::::::
    - input：任意形状、数据类型为 float32 或 float64 的 Tensor。
    - output：和 input 具有相同形状和数据类型的 Tensor。

代码示例
::::::::::

COPY-FROM: paddle.nn.Hardsigmoid
