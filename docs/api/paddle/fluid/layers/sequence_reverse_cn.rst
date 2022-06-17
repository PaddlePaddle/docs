.. _cn_api_fluid_layers_sequence_reverse:

sequence_reverse
-------------------------------

.. py:function:: paddle.fluid.layers.sequence_reverse(x, name=None)




**注意：该OP的输入只能是LoDTensor，如果您需要处理的输入是Tensor类型，请使用reverse函数（fluid.layers.** :ref:`cn_api_fluid_layers_reverse` **）。**

**该OP仅支持LoDTensor**，对于输入的LoDTensor，在每个序列（sequence）上进行反转。目前仅支持对LoD层次(LoD level)为1的LoDTensor进行反转。该OP在构建反向 :ref:`cn_api_fluid_layers_DynamicRNN` 网络时十分有用。

::

    输入x是一个LoDTensor:
        x.lod  = [[0, 2, 5]]
        x.data = [[1,  2,  3,  4],
                  [5,  6,  7,  8],
                  [9, 10, 11, 12],
                  [13,14, 15, 16],
                  [17,18, 19, 20]]
        x.shape = [5, 4]

    输出out与x具有同样的shape和LoD信息：
        out.lod  = [[0, 2, 5]]
        out.data = [[5,  6,  7,  8],
                    [1,  2,  3,  4],
                    [17,18, 19, 20],
                    [13,14, 15, 16],
                    [9, 10, 11, 12]]
        out.shape = [5, 4]


参数
::::::::::::

  - **x** (Variable) – 输入是LoD level为1的LoDTensor。目前仅支持对LoD层次(LoD level)为1的LoDTensor进行反转。数据类型为float32，float64，int8，int32或int64。
  - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。

返回
::::::::::::
输出在每个序列上反转后的LoDTensor，数据类型和输入类型一致。

返回类型
::::::::::::
Variable

代码示例
::::::::::::

COPY-FROM: paddle.fluid.layers.sequence_reverse