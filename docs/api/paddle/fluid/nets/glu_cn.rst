.. _cn_api_fluid_nets_glu:

glu
-------------------------------

.. py:function:: paddle.fluid.nets.glu(input, dim=-1)




门控线性单元 Gated Linear Units (GLU) 由 :ref:`cn_api_fluid_layers_split` ，:ref:`cn_api_fluid_layers_sigmoid` 和 :ref:`cn_api_fluid_layers_elementwise_mul` 组成。特定的，沿着给定维度将输入拆分成两个大小相同的部分，:math:`a` 和 :math:`b`，按如下方式计算：

.. math::
    GLU(a,b) = a \bigotimes \sigma (b)


参考论文：`Language Modeling with Gated Convolutional Networks <https://arxiv.org/pdf/1612.08083.pdf>`_

参数
::::::::::::

    - **input** (Variable) - 输入变量，多维 Tensor，支持的数据类型为 float32、float64 和 float16（GPU）。
    - **dim** (int) - 拆分的维度。如果 :math:`dim<0`，拆分的维为 :math:`rank(input) + dim`。默认为 -1，即最后一维。

返回
::::::::::::
 计算结果，尺寸为输入大小的一半，数据类型与输入的数据类型相同

返回类型
::::::::::::
变量（Variable）

代码示例
::::::::::::

COPY-FROM: paddle.fluid.nets.glu
