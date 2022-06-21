.. _cn_api_fluid_layers_sequence_reshape:

sequence_reshape
-------------------------------


.. py:function:: paddle.fluid.layers.sequence_reshape(input, new_dim)




**注意：该OP的输入只能是LoDTensor，如果您需要处理的输入是Tensor类型，请使用reshape函数（fluid.layers.** :ref:`cn_api_fluid_layers_reshape` **）。**

**该OP仅支持LoDTensor**，在指定 ``new_dim`` 参数下，通过序列原始长度、和原始shape计算出新的shape，以输出包含新维度（new_dim）下的LoDTensor。目前仅支持1-level LoDTensor，请确保(原长度*原维数)可以除以新的维数，且每个序列没有余数。

::

    input是一个LoDTensor:
        input.lod  = [[0, 2, 6]]
        input.data = [[1,  2], [3,  4],
                      [5,  6], [7,  8],
                      [9, 10], [11, 12]]
        input.shape = [6, 2]
    设置 new_dim = 4
    输出为LoDTensor:
        out.lod  = [[0, 1, 3]]

        out.data = [[1,  2,  3,  4],
                    [5,  6,  7,  8],
                    [9, 10, 11, 12]]
        out.shape = [3, 4]



参数
::::::::::::

    - **input** (Variable) - 维度为 :math:`[M, K]` 的二维LoDTensor，且仅支持lod_level为1。数据类型为int32，int64，float32或float64。
    - **new_dim** (int)- 指定reshape后的新维度，即对输入LoDTensor重新reshape后的新维度。

返回
::::::::::::
根据新维度重新reshape后的LoDTensor，数据类型和输入一致。

返回类型
::::::::::::
Variable

代码示例
::::::::::::

COPY-FROM: paddle.fluid.layers.sequence_reshape