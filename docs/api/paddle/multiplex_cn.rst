.. _cn_api_paddle_multiplex:

multiplex
-------------------------------

.. py:function:: paddle.multiplex(inputs, index, name=None)



根据给定的 index 参数，从每个输入 Tensor 中选择特定行构造输出 Tensor。

设输入包含 :math:`m` 个 Tensor，其中 :math:`I_{i}` 代表第 i 个输入 Tensor，:math:`i` 处于区间 :math:`[0,m)`。

设输出为 :math:`O`，其中 :math:`O[i]` 为输出的第 i 行，则输出满足：:math:`O[i] = I_{index[i]}[i]`

示例：

.. code-block:: text

        # 输入为 4 个 shape 为[4,4]的 Tensor
        inputs = [[[0,0,3,4], [0,1,3,4], [0,2,4,4], [0,3,3,4]],
                  [[1,0,3,4], [1,1,7,8], [1,2,4,2], [1,3,3,4]],
                  [[2,0,3,4], [2,1,7,8], [2,2,4,2], [2,3,3,4]],
                  [[3,0,3,4], [3,1,7,8], [3,2,4,2], [3,3,3,4]]]

        # index 为 shape 为[4,1]的 Tensor
        index = [[3],[0],[1],[2]]

        # 输出 shape 为[4,4]
        out = [[3,0,3,4]    // out[0] = inputs[index[0]][0] = inputs[3][0] = [3,0,3,4]
               [0,1,3,4]    // out[1] = inputs[index[1]][1] = inputs[0][1] = [0,1,3,4]
               [1,2,4,2]    // out[2] = inputs[index[2]][2] = inputs[1][2] = [1,2,4,2]
               [2,3,3,4]]   // out[3] = inputs[index[3]][3] = inputs[2][3] = [2,3,3,4]

参数
::::::::::::

  - **inputs** （list） - 为输入 Tensor 列表，列表元素为数据类型为 float32、float64、int32、int64 的多维 Tensor。所有输入 Tensor 的 shape 应相同，秩必须至少为 2。
  - **index** （Tensor）- 用来选择输入 Tensor 中的某些行构建输出 Tensor 的索引，为数据类型为 int32 或 int64、shape 为[M, 1]的 2-D Tensor，其中 M 为输入 Tensor 个数。
  - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。

返回
::::::::::::
Tensor，进行 Multiplex 运算后的输出 Tensor。

代码示例
::::::::::::

COPY-FROM: paddle.multiplex
