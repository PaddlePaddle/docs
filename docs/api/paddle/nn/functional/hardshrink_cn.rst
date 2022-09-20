.. _cn_api_nn_cn_hard_shrink:

hardshrink
-------------------------------
.. py:function:: paddle.nn.functional.hardshrink(x, threshold=0.5, name=None)

hardshrink 激活层。计算公式如下：

.. math::

    hardshrink(x)=
        \left\{
        \begin{aligned}
        &x, & & if \ x > threshold \\
        &x, & & if \ x < -threshold \\
        &0, & & if \ others
        \end{aligned}
        \right.

其中，:math:`x` 为输入的 Tensor

参数
::::::::::
    - **x** (Tensor) - 输入的 ``Tensor``，数据类型为：float32、float64。
    - **threshold** (float，可选) - hard_shrink 激活计算公式中的 threshold 值。默认值为 0.5。
    - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。

返回
::::::::::
    ``Tensor``，数据类型和形状同 ``x`` 一致。

代码示例
::::::::::

COPY-FROM: paddle.nn.functional.hardshrink
