.. _cn_api_fluid_layers_mish:

mish
-------------------------------

.. py:function:: paddle.fluid.layers.mish(x, threshold=20, name=None)




Mish 激活函数。参考 `Mish: A Self Regularized Non-Monotonic Neural Activation Function <https://arxiv.org/abs/1908.08681>`_ 。

.. math::
        softplus(x) = \begin{cases}
                x, \text{if } x > \text{threshold} \\
                \ln(1 + e^{x}),  \text{otherwise}
            \end{cases}

        Mish(x) = x * \tanh(softplus(x))

参数
::::::::::::

    - **x** (Variable) -  多维 Tensor，数据类型为 float32，float64。
    - **threshold** (float) - Mish 激活函数中计算 softplus 的阈值。如果输入大于该阈值，将使用近似计算，默认值为 20.0。
    - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。

返回
::::::::::::

    - Variable: Mish op 的结果，多维 Tensor。数据类型为 float32 或 float64，数据类型以及形状和输入 x 一致。

代码示例
::::::::::::

COPY-FROM: paddle.fluid.layers.mish
