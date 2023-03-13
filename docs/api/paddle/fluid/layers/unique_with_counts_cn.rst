.. _cn_api_fluid_layers_unique_with_counts:

unique_with_counts
-------------------------------

.. py:function:: paddle.fluid.layers.unique_with_counts(x, dtype='int32')



该 OP 对输入 Tensor 元素进行去重，获取去重后结果 Tensor，同时获取去重后结果在原始输入中的计数 Tensor 以及在原始输入中的索引 Tensor。

注：该 OP 仅支持 **CPU**，同时仅支持 **Tensor**

参数
::::::::::::

    - **x** (Variable) – 数据 shape 为 :math:`[N]` 的一维 Tensor，数据类型为 float32，float64，int32，int64。
    - **dtype** (np.dtype|core.VarDesc.VarType|str) – 索引和计数 Tensor 的类型，默认为 int32，数据类型需要为 int32 或 int64。

返回
::::::::::::

    - **out** 表示对输入进行去重后结果一维 Tensor，数据 shape 为 :math:`[K]` ，K 和输入 x 的 shape 中的 N 可能不一致。
    - **index** 表示原始输入在去重后结果中的索引 Tensor :math:`[N]` ，shape 和输入 x 的 shape 一致。
    - **count** 表示去重后元素的计数结果 Tensor，数据 shape 为 :math:`[K]`，数据 shape 和 out 的 shape 一致。

返回类型
::::::::::::
tuple，tuple 中元素类型为 Variable(Tensor)，输出中的 out 和输入 x 的数据类型一致，输出中 index 以及 count 的数据类型为 int32，int64。

代码示例
::::::::::::

COPY-FROM: paddle.fluid.layers.unique_with_counts
