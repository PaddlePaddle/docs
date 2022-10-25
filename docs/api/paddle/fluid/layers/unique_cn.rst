.. _cn_api_fluid_layers_unique:

unique
-------------------------------

.. py:function:: paddle.fluid.layers.unique(x, dtype='int32')

unique为 ``x`` 返回一个unique张量和一个指向该unique张量的索引。

参数
::::::::::::

    - **x** (Tensor) - 一个1维输入张量
    - **dtype** (np.dtype|str，可选) – 索引张量的类型，应该为int32或者int64。默认：int32。

返回
::::::::::::
元组(out, index)。 ``out`` 为 ``x`` 的指定dtype的unique张量，``index`` 是一个指向 ``out`` 的索引张量，用户可以通过该函数来转换原始的 ``x`` 张量的索引。

返回类型
::::::::::::
元组(tuple)

代码示例
::::::::::::

COPY-FROM: paddle.fluid.layers.unique