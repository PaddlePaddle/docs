.. _cn_api_fluid_layers_sequence_expand_as:

sequence_expand_as
-------------------------------


.. py:function:: paddle.fluid.layers.sequence_expand_as(x, y, name=None)




Sequence Expand As Layer，该 OP 根据输入 ``y`` 的第 0 级 lod 对输入 ``x`` 进行扩展。当前实现要求 ``y`` 的 lod 层数（level）必须为 1，且 ``x`` 的第一维必须和 ``y`` 的第 0 层 lod 大小相同，所以扩展后的 LodTensor 具有和 ``y`` 相同的 lod。扩展结果与输入 ``x`` 的 lod 无关，所以无需考虑 ``x`` 的 lod。

注意，该 OP 的输入 ``x`` 可以是 Tensor， ``y`` 只能是 LodTensor。

范例解释如下：

::

    例 1:
    假设，有 4 个长度维 1 的序列[a]、[b]、[c]和[d]，现在要将其扩展为长度是 3、3、1、1 的序列[a][a][a]、[b][b][b]、[c]和[d]。
    显然，扩展后的序列 lod 为[0, 3, 6, 7, 8]，则：
    给定输入一维 LoDTensor x
        x.data = [[a], [b], [c], [d]]
        x.dims = [4, 1]
    和输入 y
        y.lod = [[3, 3, 1, 1]]    #为了便于理解这里用基于长度 lod 表示

    经过 sequence_expand_as 运算，得到输出 1 级 LoDTensor out
        out.lod =  [[0,            3,              6,  7,  8]]    #基于偏移的 lod，等价于基于长度的[[3, 3, 1, 1]]
        out.data = [[a], [a], [a], [b], [b], [b], [c], [d]]
        out.dims = [8, 1]

    可见，输出 out 将 x 扩展至和 y 具有相同的 lod。

::

    例 2：
    设定与例 1 类似，给定输入一维 LoDTensor x：
        x.data = [[a, b], [c, d], [e, f]]
        x.dims = [3, 2]
    和输入 y：
        y.lod = [[2, 1, 3]]    #为了便于理解这里用基于长度 lod 表示

    输出为 1 级 LoDTensor：
        out.lod =  [[0,             2,     3,                    6]]    #基于偏移的 lod，等价于基于长度的[[2, 1, 3]]
        out.data = [[a, b], [a, b] [c, d], [e, f], [e, f], [e, f]]
        out.dims = [6, 2]

    可见，输出 out 将 x 扩展至和 y 具有相同的 lod。


参数
::::::::::::

    - **x** (Variable) - 输入变量，维度为 :math:`[M, K]` 的二维 Tensor，第一维必须与输入 ``y`` 的第 0 层 lod 大小相同，且仅支持 lod_level 为 1。数据类型支持 int32，int64，float32 或 float64。
    - **y** (Variable) - 输入变量，LoDTensor，lod level 必须为 1。
    - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。

返回
::::::::::::
扩展变量，维度为 :math:`[N, K]` 的二维 LoDTensor，N 由输入 ``y`` 的 lod 决定，且仅支持 lod_level 为 1。数据类型与输入 ``x`` 一致。

返回类型
::::::::::::
Variable


代码示例
::::::::::::

COPY-FROM: paddle.fluid.layers.sequence_expand_as
