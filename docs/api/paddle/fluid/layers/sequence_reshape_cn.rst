.. _cn_api_fluid_layers_sequence_reshape:

sequence_reshape
-------------------------------


.. py:function:: paddle.fluid.layers.sequence_reshape(input, new_dim)




**注意：该 OP 的输入只能是 LoDTensor，如果您需要处理的输入是 Tensor 类型，请使用 reshape 函数（fluid.layers.** :ref:`cn_api_fluid_layers_reshape` **）。**

**该 OP 仅支持 LoDTensor**，在指定 ``new_dim`` 参数下，通过序列原始长度、和原始 shape 计算出新的 shape，以输出包含新维度（new_dim）下的 LoDTensor。目前仅支持 1-level LoDTensor，请确保(原长度*原维数)可以除以新的维数，且每个序列没有余数。

::

    input 是一个 LoDTensor:
        input.lod  = [[0, 2, 6]]
        input.data = [[1,  2], [3,  4],
                      [5,  6], [7,  8],
                      [9, 10], [11, 12]]
        input.shape = [6, 2]
    设置 new_dim = 4
    输出为 LoDTensor:
        out.lod  = [[0, 1, 3]]

        out.data = [[1,  2,  3,  4],
                    [5,  6,  7,  8],
                    [9, 10, 11, 12]]
        out.shape = [3, 4]



参数
::::::::::::

    - **input** (Variable) - 维度为 :math:`[M, K]` 的二维 LoDTensor，且仅支持 lod_level 为 1。数据类型为 int32，int64，float32 或 float64。
    - **new_dim** (int)- 指定 reshape 后的新维度，即对输入 LoDTensor 重新 reshape 后的新维度。

返回
::::::::::::
根据新维度重新 reshape 后的 LoDTensor，数据类型和输入一致。

返回类型
::::::::::::
Variable

代码示例
::::::::::::

COPY-FROM: paddle.fluid.layers.sequence_reshape
