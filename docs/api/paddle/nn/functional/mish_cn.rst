.. _cn_api_paddle_nn_functional_mish:

mish
-------------------------------

.. py:function:: paddle.nn.functional.mish(x, name=None)

mish 激活层。计算公式如下：

.. math::

        softplus(x) = \begin{cases}
                x, \text{if } x > \text{threshold} \\
                \ln(1 + e^{x}),  \text{otherwise}
            \end{cases}

        Mish(x) = x * \tanh(softplus(x))


参数
::::::::::
    - **x** (Tensor) - 输入的 ``Tensor``，数据类型为：float32、float64。
    - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。

返回
::::::::::
    ``Tensor``，数据类型和形状同 ``x`` 一致。

代码示例
::::::::::

COPY-FROM: paddle.nn.functional.mish
