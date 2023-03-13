.. _cn_api_fluid_layers_zeros_like:

zeros_like
-------------------------------

.. py:function:: paddle.fluid.layers.zeros_like(x, out=None)





该 OP 创建一个和 x 具有相同的形状和数据类型的全零 Tensor。

参数
::::::::::::

    - **x** (Variable) – 指定输入为一个多维的 Tensor，数据类型可以是 bool，float32，float64，int32，int64。
    - **out** (Variable|可选) – 如果为 None，则创建一个 Variable 作为输出，创建后的 Variable 的数据类型，shape 大小和输入变量 x 一致。如果是输入的一个 Tensor，数据类型和数据 shape 大小需要和输入变量 x 一致。默认值为 None。

返回
::::::::::::
返回一个多维的 Tensor，具体的元素值和输入的数据类型相关，如果是 bool 类型的，则全 False，其它均为 0。数据 shape 大小和输入 x 一致。

返回类型
::::::::::::
Variable

代码示例
::::::::::::

COPY-FROM: paddle.fluid.layers.zeros_like
