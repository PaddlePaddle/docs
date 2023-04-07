.. _cn_api_nn_cn_hardtanh:

hardtanh
-------------------------------
.. py:function:: paddle.nn.functional.hardtanh(x, min=-1.0, max=1.0, name=None):

hardtanh 激活层（Hardtanh Activation Operator）。计算输入 `x` 的 `hardtanh` 值。计算公式如下：

.. math::

    hardtanh(x)=
        \left\{
        \begin{aligned}
        &max, & & if \ x > max \\
        &min, & & if \ x < min \\
        &x, & & if \ others
        \end{aligned}
        \right.

其中，:math:`x` 为输入的 Tensor。

参数
::::::::::
    - **x** (Tensor) - 输入的 ``Tensor``，数据类型为：float16、float32、float64、uint16。
    - **min** (float，可选) - hardtanh 激活计算公式中的 min 值。默认值为-1。
    - **max** (float，可选) - hardtanh 激活计算公式中的 max 值。默认值为 1。
    - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。

返回
::::::::::
    ``Tensor``，数据类型和形状同 ``x`` 一致。

代码示例
:::::::::

COPY-FROM: paddle.nn.functional.hardtanh
