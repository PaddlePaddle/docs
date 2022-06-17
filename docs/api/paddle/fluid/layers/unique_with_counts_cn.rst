.. _cn_api_fluid_layers_unique_with_counts:

unique_with_counts
-------------------------------

.. py:function:: paddle.fluid.layers.unique_with_counts(x, dtype='int32')



该OP对输入Tensor元素进行去重，获取去重后结果Tensor，同时获取去重后结果在原始输入中的计数Tensor以及在原始输入中的索引Tensor。

注：该OP仅支持 **CPU**，同时仅支持 **Tensor**

参数
::::::::::::

    - **x** (Variable) – 数据shape为 :math:`[N]` 的一维Tensor，数据类型为 float32，float64，int32，int64。
    - **dtype** (np.dtype|core.VarDesc.VarType|str) – 索引和计数Tensor的类型，默认为 int32，数据类型需要为 int32或int64。

返回
::::::::::::
 
    - **out** 表示对输入进行去重后结果一维Tensor，数据shape为 :math:`[K]` ，K和输入x的shape中的N可能不一致。
    - **index** 表示原始输入在去重后结果中的索引Tensor :math:`[N]` ，shape和输入x的shape一致。
    - **count** 表示去重后元素的计数结果Tensor，数据shape为 :math:`[K]`，数据shape和out的shape一致。

返回类型
::::::::::::
tuple，tuple中元素类型为Variable(Tensor)，输出中的out和输入x的数据类型一致，输出中index以及count的数据类型为 int32，int64。

代码示例
::::::::::::

COPY-FROM: paddle.fluid.layers.unique_with_counts