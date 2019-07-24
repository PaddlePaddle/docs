.. _cn_api_fluid_layers_sequence_reshape:

sequence_reshape
-------------------------------

.. py:function:: paddle.fluid.layers.sequence_reshape(input, new_dim)

Sequence Reshape Layer
该层重排输入序列。用户设置新维度。每一个序列的的长度通过原始长度、原始维度和新的维度计算得出。以下实例帮助解释该层的功能

.. code-block:: python

    x是一个LoDTensor:
        x.lod  = [[0, 2, 6]]
        x.data = [[1,  2], [3,  4],
                [5,  6], [7,  8],
                [9, 10], [11, 12]]
        x.dims = [6, 2]
    设置 new_dim = 4
    输出为LoDTensor:
        out.lod  = [[0, 1, 3]]

        out.data = [[1,  2,  3,  4],
                    [5,  6,  7,  8],
                    [9, 10, 11, 12]]
        out.dims = [3, 4]

目前仅提供1-level LoDTensor，请确保(原长度*原维数)可以除以新的维数，每个序列没有余数。

参数：
    - **input** (Variable)-一个2-D LoDTensor,模型为[N,M]，维度为M
    - **new_dim** (int)-新维度，输入LoDTensor重新塑造后的新维度

返回：根据新维度重新塑造的LoDTensor

返回类型：变量（Variable）

**代码示例**：

.. code-block:: python

    import paddle.fluid as fluid
    x = fluid.layers.data(name='x', shape=[2, 6], append_batch_size=False, dtype='float32', lod_level=1)
    x_reshaped = fluid.layers.sequence_reshape(input=x, new_dim=4)









