.. _cn_api_nn_SELU:

SELU
-------------------------------
.. py:class:: paddle.nn.SELU(scale=1.0507009873554804934193349852946, alpha=1.6732632423543772848170429916717, name=None)

SELU 激活层

.. math::

    SELU(x)= scale *
             \begin{cases}
               x, \text{if } x > 0 \\
               alpha * e^{x} - alpha, \text{if } x <= 0
             \end{cases}

其中，:math:`x` 为输入的 Tensor。

参数
::::::::::
    - **scale** (float，可选) - SELU 激活计算公式中的 scale 值，必须大于 1.0。默认值为 1.0507009873554804934193349852946。
    - **alpha** (float，可选) - SELU 激活计算公式中的 alpha 值，必须大于等于零。默认值为 1.6732632423543772848170429916717。
    - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。

形状
::::::::::
    - input：任意形状的 Tensor。
    - output：和 input 具有相同形状的 Tensor。

代码示例
:::::::::

COPY-FROM: paddle.nn.SELU
