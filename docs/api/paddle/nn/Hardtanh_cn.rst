.. _cn_api_nn_Hardtanh:

Hardtanh
-------------------------------
.. py:class:: paddle.nn.Hardtanh(min=-1.0, max=1.0, name=None)

Hardtanh 激活层（Hardtanh Activation Operator）。创建一个 `Hardtanh` 类的可调用对象。计算公式如下：

.. math::

    Hardtanh(x)=
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
    - **min** (float，可选) - Hardtanh 激活计算公式中的 min 值。默认值为-1。
    - **max** (float，可选) - Hardtanh 激活计算公式中的 max 值。默认值为 1。
    - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。

形状
::::::::::
    - input：任意形状的 Tensor。
    - output：和 input 具有相同形状的 Tensor。

代码示例
:::::::::

COPY-FROM: paddle.nn.Hardtanh
