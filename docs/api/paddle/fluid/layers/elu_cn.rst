.. _cn_api_fluid_layers_elu:

elu
-------------------------------

.. py:function:: paddle.fluid.layers.elu(x, alpha=1.0, name=None)




ELU 激活层（ELU Activation Operator）

根据 https://arxiv.org/abs/1511.07289 对输入 Tensor 中每个元素应用以下计算。

.. math::
        \\out=max(0,x)+min(0,α∗(e^{x}−1))\\

参数
::::::::::::

 - **x** (Variable) - 该 OP 的输入为多维 Tensor。数据类型为 float32 或 float64。
 - **alpha** (float，可选) - ELU 的 alpha 值，默认值为 1.0。
 - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。

返回
::::::::::::
 输出为 Tensor，与 ``x`` 维度相同、数据类型相同。

返回类型
::::::::::::
 Variable

代码示例
::::::::::::

COPY-FROM: paddle.fluid.layers.elu
