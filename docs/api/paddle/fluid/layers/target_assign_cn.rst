.. _cn_api_fluid_layers_target_assign:

target_assign
-------------------------------

.. py:function:: paddle.fluid.layers.target_assign(input, matched_indices, negative_indices=None, mismatch_value=None, name=None)




对于每个实例，根据 ``match_indices`` 和 ``negative_indices`` 位置索引，给输入 ``out`` 和 ``out_weight`` 赋值。输入 ``input`` 和 ``negative_indices`` 均为 2-D LoDTensor。假如 ``input`` 中每个实例的行偏移称作 lod，该操作计算步骤如下：

1. 根据 match_indices 赋值：

.. code-block:: text

    If id = match_indices[i][j] > 0,

        out[i][j][0 : K] = X[lod[i] + id][j % P][0 : K]
        out_weight[i][j] = 1.

    Otherwise,

        out[j][j][0 : K] = {mismatch_value, mismatch_value, ...}
        out_weight[i][j] = 0.

2. 如果提供 neg_indices，则再次依据该输入赋值：

neg_indices 中的第 i 个实例的索引称作 neg_indice，则对于第 i 个实例：

.. code-block:: text

    for id in neg_indice:
        out[i][id][0 : K] = {mismatch_value, mismatch_value, ...}
        out_weight[i][id] = 1.0

参数
::::::::::::

    - **input** (Variable) - 输入为 3-D LoDTensor，为了方便在上述文档中解释，假如维度是[M,P,K]。
    - **matched_indices** (Variable) - 输入为 2-D Tensor，数据类型为 int32，表示在输入中匹配位置，具体计算如上，同样，为了方便解释，假如维度大小为[N,P]，如果 ``matched_indices[i][j]`` 为-1，表示在第 ``i`` 个实例中第 j 列项没有任何匹配项，输出会设置成 ``mismatch_value`` 。
    - **negative_indices** (Variable，可选) - 维度为 2-D LoDTensor，数据类型为 int32。可以不设置，如果设置，会依据该位置索引再次给输出赋值，具体参考上述文档。
    - **mismatch_value** (float32，可选) - 未匹配的位置填充值。
    - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。

返回
::::::::::::
返回一个元组（out,out_weight）。out 是三维 Tensor，维度为[N,P,K]，N 和 P 与 ``matched_indices`` 中的 N 和 P 一致，K 和输入 X 中的 K 一致。``out_weight`` 的维度为[N,P,1]。

返回类型
::::::::::::
tuple(Variable)

代码示例
::::::::::::

.. code-block:: python

        import paddle.fluid as fluid
        x = fluid.data(
            name='x',
            shape=[4, 20, 4],
            dtype='float',
            lod_level=1)
        matched_id = fluid.data(
            name='indices',
            shape=[8, 20],
            dtype='int32')
        trg, trg_weight = fluid.layers.target_assign(
            x,
            matched_id,
            mismatch_value=0)
