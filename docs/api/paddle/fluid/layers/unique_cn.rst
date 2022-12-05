.. _cn_api_fluid_layers_unique:

unique
-------------------------------

.. py:function:: paddle.fluid.layers.unique(x, dtype='int32')

unique 为 ``x`` 返回一个 uniqueTensor 和一个指向该 uniqueTensor 的索引。

参数
::::::::::::

    - **x** (Tensor) - 一个 1 维输入 Tensor
    - **dtype** (np.dtype|str，可选) – 索引 Tensor 的类型，应该为 int32 或者 int64。默认：int32。

返回
::::::::::::
元组(out, index)。 ``out`` 为 ``x`` 的指定 dtype 的 uniqueTensor，``index`` 是一个指向 ``out`` 的索引 Tensor，用户可以通过该函数来转换原始的 ``x`` Tensor 的索引。

返回类型
::::::::::::
元组(tuple)

代码示例
::::::::::::

COPY-FROM: paddle.fluid.layers.unique
