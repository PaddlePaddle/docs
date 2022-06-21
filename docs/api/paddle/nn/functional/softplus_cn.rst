.. _cn_api_nn_cn_softplus:

softplus
-------------------------------

.. py:function:: paddle.nn.functional.softplus(x, beta=1, threshold=20, name=None)

softplus激活层

.. math::

    softplus(x) = \frac{1}{beta} * \log(1 + e^{beta * x}) \\
    \text{为了保证数值稳定性，当}\,beta * x > threshold\,\text{时，函数转变为线性函数x}。

其中，:math:`x` 为输入的 Tensor

参数
::::::::::::

::::::::::
 - x (Tensor) - 输入的 ``Tensor``，数据类型为：float32、float64。
 - beta (float，可选) - Softplus激活计算公式中的beta值。默认值为1。
 - threshold (float，可选) - Softplus激活计算公式中的threshold值。默认值为20。
 - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。

返回
::::::::::
    ``Tensor``，数据类型和形状同 ``x`` 一致。

代码示例
::::::::::

COPY-FROM: paddle.nn.functional.softplus