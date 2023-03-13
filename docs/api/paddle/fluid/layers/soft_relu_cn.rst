.. _cn_api_fluid_layers_soft_relu:

soft_relu
-------------------------------

.. py:function:: paddle.fluid.layers.soft_relu(x, threshold=40.0, name=None)




SoftReLU 激活函数。

.. math::   out=ln(1+exp(max(min(x,threshold),-threshold)))

参数
::::::::::::

    - **x** (Variable) - SoftReLU 激活函数的输入，为数据类型为 float32，float64 的多维 Tensor 或者 LoDTensor。
    - **threshold** (float) - SoftRelu 的阈值，默认为 40.0。
    - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。

返回
::::::::::::
一个 Tensor，shape 和输入 Tensor 相同。

返回类型
::::::::::::
Variable(Tensor|LoDTensor)，LoD 信息与输入 Tensor 一致。

代码示例
::::::::::::

COPY-FROM: paddle.fluid.layers.soft_relu
