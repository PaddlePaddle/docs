.. _cn_api_nn_cn_leaky_relu:

leaky_relu
-------------------------------
.. py:function:: paddle.nn.functional.leaky_relu(x, negative_slope=0.01, name=None)

leaky_relu 激活层。计算公式如下：

.. math::

    LeakyReLU(x)=
        \left\{
        \begin{aligned}
        &x, & & if \ x >= 0 \\
        &negative\_slope * x, & & otherwise \\
        \end{aligned}
        \right. \\

其中，:math:`x` 为输入的 Tensor。

参数
::::::::::
    - **x** (Tensor) - 输入的 Tensor，数据类型为：float32、float64。
    - **negative_slope** (float，可选) - :math:`x < 0` 时的斜率。默认值为 0.01。
    - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。

返回
::::::::::
    ``Tensor``，数据类型和形状同 ``x`` 一致。

代码示例
::::::::::

COPY-FROM: paddle.nn.functional.leaky_relu
