.. _cn_api_fluid_layers_leaky_relu:

leaky_relu
-------------------------------

.. py:function:: paddle.fluid.layers.leaky_relu(x, alpha=0.02, name=None)




LeakyRelu激活函数

.. math::   out=max(x,α∗x)

参数
::::::::::::

    - **x** (Variable) - 输入的多维LoDTensor/Tensor，数据类型为：float32，float64。
    - **alpha** (float) - 负斜率，缺省值为0.02。
    - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。

返回
::::::::::::
 与 ``x`` 维度相同，数据类型相同的LodTensor/Tensor。

返回类型
::::::::::::
 Variable

代码示例
::::::::::::

COPY-FROM: paddle.fluid.layers.leaky_relu