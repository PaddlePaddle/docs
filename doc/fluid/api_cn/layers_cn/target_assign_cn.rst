.. _cn_api_fluid_layers_target_assign:

target_assign
-------------------------------

.. py:function:: paddle.fluid.layers.target_assign(input, matched_indices, negative_indices=None, mismatch_value=None, name=None)

对于给定的目标边界框（bounding box）和标签（label），该操作符对每个预测赋予分类和逻辑回归目标函数以及预测权重。权重具体表示哪个预测无需贡献训练误差。

对于每个实例，根据 ``match_indices`` 和 ``negative_indices`` 赋予输入 ``out`` 和 ``out_weight``。将定输入中每个实例的行偏移称为lod，该操作符执行分类或回归目标函数，执行步骤如下：

1.根据match_indices分配所有输入

.. code-block:: text

    If id = match_indices[i][j] > 0,

        out[i][j][0 : K] = X[lod[i] + id][j % P][0 : K]
        out_weight[i][j] = 1.

    Otherwise,

        out[j][j][0 : K] = {mismatch_value, mismatch_value, ...}
        out_weight[i][j] = 0.

2.如果提供neg_indices，根据neg_indices分配out_weight：

假设neg_indices中每个实例的行偏移称为neg_lod，该实例中第i个实例和neg_indices的每个id如下：

.. code-block:: text

    out[i][id][0 : K] = {mismatch_value, mismatch_value, ...}
    out_weight[i][id] = 1.0

参数：
    - **inputs** (Variable) - 输入为三维LoDTensor，维度为[M,P,K]
    - **matched_indices** (Variable) - 张量（Tensor），整型，输入匹配索引为二维张量（Tensor），类型为整型32位，维度为[N,P]，如果MatchIndices[i][j]为-1，在第i个实例中第j列项不匹配任何行项。
    - **negative_indices** (Variable) - 输入负例索引，可选输入，维度为[Neg,1]，类型为整型32，Neg为负例索引的总数
    - **mismatch_value** (float32) - 为未匹配的位置填充值

返回：返回一个元组（out,out_weight）。out是三维张量，维度为[N,P,K],N和P与neg_indices中的N和P一致，K和输入X中的K一致。如果match_indices[i][j]存在，out_weight是输出权重，维度为[N,P,1]。

返回类型：元组（tuple）

**代码示例**：

.. code-block:: python

        import paddle.fluid as fluid
        x = fluid.layers.data(
            name='x',
            shape=[4, 20, 4],
            dtype='float',
            lod_level=1,
            append_batch_size=False)
        matched_id = fluid.layers.data(
            name='indices',
            shape=[8, 20],
            dtype='int32',
            append_batch_size=False)
        trg, trg_weight = fluid.layers.target_assign(
            x,
            matched_id,
            mismatch_value=0)






