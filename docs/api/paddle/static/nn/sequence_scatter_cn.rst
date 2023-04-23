.. _cn_api_fluid_layers_sequence_scatter:

sequence_scatter
-------------------------------


.. py:function:: paddle.static.nn.sequence_scatter(input, index, updates, name=None)


.. note::
    该 OP 的输入 index，updates 必须是带有 LoD 信息的 Tensor。

根据 index 提供的位置将 updates 中的信息更新到输出中。

先使用 input 初始化 output，然后通过 output[instance_index][index[pos]] += updates[pos]方式，将 updates 的信息更新到 output 中，其中 instance_idx 是 pos 对应的在 batch 中第 k 个样本。

output[i][j]的值取决于能否在 index 中第 i+1 个区间中找到对应的数据 j，若能找到 out[i][j] = input[i][j] + update[m][n]，否则 out[i][j] = input[i][j]。

例如，在下面样例中，index 的 lod 信息分为了 3 个区间。其中，out[0][0]能在 index 中第 1 个区间中找到对应数据 0，所以，使用 updates 对应位置的值进行更新，out[0][0] = input[0][0]+updates[0][0]。out[2][1]不能在 index 中第 3 个区间找到对应数据 1，所以，它等于输入对应位置的值，out[2][1] = input[2][1]。

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
:::::::::
      - **input** (Tensor) - 维度为 :math:`[N, k_1 ... k_n]` 的 Tensor，支持的数据类型：float32，float64，int32，int64。
      - **index** (Tensor) - 包含 index 信息的 Tensor，lod level 必须等于 1，支持的数据类型：int32，int64。
      - **updates** (Tensor) - 包含 updates 信息的 Tensor，lod level 和 index 一致，数据类型与 input 的数据类型一致。支持的数据类型：float32，float64，int32，int64。
      - **name**  (str，可选) – 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。

返回
:::::::::
在 input 的基础上使用 updates 进行更新后得到的 Tensor，它与 input 有相同的维度和数据类型。


代码示例
:::::::::
COPY-FROM: paddle.static.nn.sequence_scatter
