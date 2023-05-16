.. _cn_api_nn_Hardswish:

Hardswish
-------------------------------

.. py:function:: paddle.nn.Hardswish(name=None)

Hardswish 激活函数。创建一个 `Hardswish` 类的可调用对象。在 MobileNetV3 架构中被提出，相较于 :ref:`cn_api_nn_swish` 函数，具有数值稳定性好，计算速度快等优点，具体原理请参考：`Searching for MobileNetV3 <https://arxiv.org/pdf/1905.02244.pdf>`_ 。

.. math::

    hardswish(x)=
        \left\{
        \begin{aligned}
        &0, & & \text{if } x \leq -3 \\
        &x, & & \text{if } x \geq 3 \\
        &\frac{x(x+3)}{6}, & & \text{otherwise}
        \end{aligned}
        \right.

其中，:math:`x` 为输入的 Tensor。

参数

::::::::::
    - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。

形状
::::::::::
    - input：任意形状的 Tensor。
    - output：和 input 具有相同形状的 Tensor。

代码示例
::::::::::

COPY-FROM: paddle.nn.Hardswish
