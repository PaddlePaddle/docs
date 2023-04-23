.. _cn_api_fluid_layers_zeros_like:

zeros_like
-------------------------------

.. py:function:: paddle.fluid.layers.zeros_like(x, out=None)





该OP创建一个和x具有相同的形状和数据类型的全零Tensor。

参数
::::::::::::

    - **x** (Variable) – 指定输入为一个多维的Tensor，数据类型可以是bool，float32，float64，int32，int64。
    - **out** (Variable|可选) – 如果为None，则创建一个Variable作为输出，创建后的Variable的数据类型，shape大小和输入变量x一致。如果是输入的一个Tensor，数据类型和数据shape大小需要和输入变量x一致。默认值为None。
    
返回
::::::::::::
返回一个多维的Tensor，具体的元素值和输入的数据类型相关，如果是bool类型的，则全False，其它均为0。数据shape大小和输入x一致。

返回类型
::::::::::::
Variable

代码示例
::::::::::::

COPY-FROM: paddle.fluid.layers.zeros_like