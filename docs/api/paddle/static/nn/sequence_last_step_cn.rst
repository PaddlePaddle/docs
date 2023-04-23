.. _cn_api_fluid_layers_sequence_last_step:

sequence_last_step
-------------------------------


.. py:function:: paddle.static.nn.sequence_last_step(input)


.. note::
该 API 仅支持带有 LoD 信息的 Tensor 类型的输入。

对输入的 Tensor，在最后一层 lod_level 上，选取其每个序列（sequence）的最后一个时间步（time-step）的特征向量作为池化后的输出向量。

::

    Case 1:

        input 是 1-level 的 Tensor:
            input.lod = [[0, 2, 5, 7]]
            input.data = [[1.], [3.], [2.], [4.], [6.], [5.], [1.]]
            input.shape = [7, 1]

        输出为 Tensor:
            out.shape = [3, 1]
            且 out.shape[0] == len(x.lod[-1]) == 3

            out.data = [[3.], [6.], [1.]], where 3.=last(1., 3.), 6.=last(2., 4., 6.), 1.=last(5., 1.)

    Case 2:

        input 是 2-level 的 Tensor，包含 3 个长度分别为[2, 0, 3]的序列，其中中间的 0 表示序列为空。
        第一个长度为 2 的序列包含 2 个长度分别为[1, 2]的子序列；
        最后一个长度为 3 的序列包含 3 个长度分别为[1, 0, 3]的子序列。
            input.lod = [[0, 2, 2, 5], [0, 1, 3, 4, 4, 7]]
            input.data = [[1.], [3.], [2.], [4.], [6.], [5.], [1.]]
            input.shape = [7, 1]

        将根据最后一层的 lod 信息[0, 1, 3, 4, 4, 7]进行池化操作，且 pad_value = 0.0
        输出为 Tensor：
            out.shape= [5, 1]
            out.lod = [[0, 2, 2, 5]]
            其中 out.shape[0] == len(x.lod[-1]) == 5
            out.data = [[1.], [2.], [4.], [0.0], [1.]]
            where 1.=last(1.), 2.=last(3., 2.), 4.=last(4.), 0.0 = pad_value, 1=last(6., 5., 1.)

参数
:::::::::
**input** (Tensor)- 类型为 Tensor 的输入序列，仅支持 lod_level 不超过 2 的 Tensor，数据类型为 float32。

返回
:::::::::
每个输入序列中的最后一步特征向量组成的 Tensor，数据类型为 float32。

代码示例
:::::::::
COPY-FROM: paddle.static.nn.sequence_last_step
