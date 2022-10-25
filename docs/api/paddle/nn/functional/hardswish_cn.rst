.. _cn_api_nn_cn_hardswish:

hardswish
-------------------------------

.. py:function:: paddle.nn.functional.hardswish(x, name=None)

hardswish 激活函数。在 MobileNetV3 架构中被提出，相较于 swish 函数，具有数值稳定性好，计算速度快等优点，具体原理请参考：https://arxiv.org/pdf/1905.02244.pdf

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
    - **x** (Tensor) - 输入的 ``Tensor``，数据类型为：float32、float64。
    - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。

返回
::::::::::
    ``Tensor``，数据类型和形状同 ``x`` 一致。

代码示例
::::::::::

COPY-FROM: paddle.nn.functional.hardswish
