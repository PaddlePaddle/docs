.. _cn_api_fluid_layers_sequence_reverse:

sequence_reverse
-------------------------------

.. py:function:: paddle.static.nn.sequence_reverse(x, name=None)


.. note::
该 API 仅支持带有 LoD 信息的 Tensor。

输入的 Tensor，在每个序列（sequence）上进行反转。目前仅支持对 LoD 层次(LoD level)为 1 的 Tensor 进行反转。该 OP 在构建反向 :ref:`cn_api_fluid_layers_DynamicRNN` 网络时十分有用。

::

    输入 x 是一个 Tensor:
        x.lod  = [[0, 2, 5]]
        x.data = [[1,  2,  3,  4],
                  [5,  6,  7,  8],
                  [9, 10, 11, 12],
                  [13,14, 15, 16],
                  [17,18, 19, 20]]
        x.shape = [5, 4]

    输出 out 与 x 具有同样的 shape 和 LoD 信息：
        out.lod  = [[0, 2, 5]]
        out.data = [[5,  6,  7,  8],
                    [1,  2,  3,  4],
                    [17,18, 19, 20],
                    [13,14, 15, 16],
                    [9, 10, 11, 12]]
        out.shape = [5, 4]


参数
:::::::::

  - **x** (Variable) - 输入是 LoD level 为 1 的 Tensor。目前仅支持对 LoD 层次(LoD level)为 1 的 Tensor 进行反转。数据类型为 float32，float64，int8，int32 或 int64。
  - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。

返回
:::::::::
输出在每个序列上反转后的 Tensor，数据类型和输入类型一致。

代码示例
::::::::::::

COPY-FROM: paddle.static.nn.sequence_reverse
