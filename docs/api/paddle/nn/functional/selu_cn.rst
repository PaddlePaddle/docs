.. _cn_api_paddle_nn_functional_selu:

selu
-------------------------------

.. py:function:: paddle.nn.functional.selu(x, scale=1.0507009873554804934193349852946, alpha=1.6732632423543772848170429916717, name=None)

selu 激活层

.. math::

    selu(x)= scale *
             \begin{cases}
               x, \text{if } x > 0 \\
               alpha * e^{x} - alpha, \text{if } x <= 0
             \end{cases}

其中，:math:`x` 为输入的 Tensor。

参数
::::::::::::

::::::::::
 - **x** (Tensor) - 输入的 ``Tensor``，数据类型为：float32、float64。
 - **scale** (float，可选) - selu 激活计算公式中的 scale 值，必须大于 1.0。默认值为 1.0507009873554804934193349852946。
 - **alpha** (float，可选) - selu 激活计算公式中的 alpha 值，必须大于等于零。默认值为 1.6732632423543772848170429916717。
 - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。

返回
::::::::::
    ``Tensor``，数据类型和形状同 ``x`` 一致。

代码示例
::::::::::

COPY-FROM: paddle.nn.functional.selu
