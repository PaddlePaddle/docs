.. _cn_api_paddle_nn_functional_hardsigmoid:

hardsigmoid
-------------------------------

.. py:function:: paddle.nn.functional.hardsigmoid(x, slope=0.1666667, offset=0.5, name=None)

hardsigmoid 激活层。sigmoid 的分段线性逼近激活函数，速度比 sigmoid 快，详细解释参见 `Noisy Activation Functions <https://arxiv.org/abs/1603.00391>`_。

.. math::

    hardsigmoid(x)=
        \left\{
        \begin{aligned}
        &0, & & \text{if } x \leq -3 \\
        &1, & & \text{if } x \geq 3 \\
        &slope * x + offset, & & \text{otherwise}
        \end{aligned}
        \right.

其中，:math:`x` 为输入的 Tensor。

参数
::::::::::::

::::::::::
    - **x** (Tensor) - 输入的 ``Tensor``，数据类型为：float32、float64。
    - **slope** (float，可选) - hardsigmoid 的斜率。默认值为 0.1666667。
    - **offset** (float，可选) - hardsigmoid 的截距。默认值为 0.5。
    - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。

返回
::::::::::
    ``Tensor``，数据类型和形状同 ``x`` 一致。

代码示例
::::::::::

COPY-FROM: paddle.nn.functional.hardsigmoid
