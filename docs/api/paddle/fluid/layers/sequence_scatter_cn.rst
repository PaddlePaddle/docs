.. _cn_api_fluid_layers_sequence_scatter:

sequence_scatter
-------------------------------


.. py:function:: paddle.fluid.layers.sequence_scatter(input, index, updates, name=None)




.. note::
    该OP的输入index，updates必须是LoDTensor。

该OP根据index提供的位置将updates中的信息更新到输出中。

该OP先使用input初始化output，然后通过output[instance_index][index[pos]] += updates[pos]方式，将updates的信息更新到output中，其中instance_idx是pos对应的在batch中第k个样本。

output[i][j]的值取决于能否在index中第i+1个区间中找到对应的数据j，若能找到out[i][j] = input[i][j] + update[m][n]，否则 out[i][j] = input[i][j]。

例如，在下面样例中，index的lod信息分为了3个区间。其中，out[0][0]能在index中第1个区间中找到对应数据0，所以，使用updates对应位置的值进行更新，out[0][0] = input[0][0]+updates[0][0]。out[2][1]不能在index中第3个区间找到对应数据1，所以，它等于输入对应位置的值，out[2][1] = input[2][1]。

**样例**:

::

    输入：

    input.data = [[1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                  [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                  [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]]
    input.dims = [3, 6]

    index.data = [[0], [1], [2], [5], [4], [3], [2], [1], [3], [2], [5], [4]]
    index.lod =  [[0,        3,                       8,                 12]]

    updates.data = [[0.3], [0.3], [0.4], [0.1], [0.2], [0.3], [0.4], [0.0], [0.2], [0.3], [0.1], [0.4]]
    updates.lod =  [[  0,            3,                                 8,                         12]]

    输出：

    out.data = [[1.3, 1.3, 1.4, 1.0, 1.0, 1.0],
                [1.0, 1.0, 1.4, 1.3, 1.2, 1.1],
                [1.0, 1.0, 1.3, 1.2, 1.4, 1.1]]
    out.dims = X.dims = [3, 6]


参数
::::::::::::

      - **input** (Variable) - 维度为 :math:`[N, k_1 ... k_n]` 的Tensor，支持的数据类型：float32，float64，int32，int64。
      - **index** (Variable) - 包含index信息的LoDTensor，lod level必须等于1，支持的数据类型：int32，int64。
      - **updates** (Variable) - 包含updates信息的LoDTensor，lod level和index一致，数据类型与input的数据类型一致。支持的数据类型：float32，float64，int32，int64。 
      - **name**  (str，可选) – 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为None。

返回
::::::::::::
在input的基础上使用updates进行更新后得到的Tensor，它与input有相同的维度和数据类型。

返回类型
::::::::::::
Variable


代码示例
::::::::::::

COPY-FROM: paddle.fluid.layers.sequence_scatter