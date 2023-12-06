.. _cn_api_paddle_static_nn_sequence_concat:

sequence_concat
-------------------------------


.. py:function:: paddle.static.nn.sequence_concat(input, name=None)

.. note::
该 OP 的输入只能是带有 LoD 信息的 Tensor，如果您需要处理的输入是一般的 Tensor 类型，请使用 :ref:`paddle.concat <cn_api_paddle_concat>` 。

**该 OP 仅支持带有 LoD 信息的 Tensor**，通过 LoD 信息将输入的多个 Tensor 进行连接（concat），输出连接后的 Tensor。

::

    input 是由多个带有 LoD 信息的 Tensor 组成的 list：
        input = [x1, x2]
    其中：
        x1.lod = [[0, 3, 5]]
        x1.data = [[1], [2], [3], [4], [5]]
        x1.shape = [5, 1]

        x2.lod = [[0, 2, 4]]
        x2.data = [[6], [7], [8], [9]]
        x2.shape = [4, 1]
    且必须满足：len(x1.lod[0]) == len(x2.lod[0])

    输出为 Tensor：
        out.lod = [[0, 3+2, 5+4]]
        out.data = [[1], [2], [3], [6], [7], [4], [5], [8], [9]]
        out.shape = [9, 1]


参数
:::::::::

        - **input** (list of Variable) – 多个带有 LoD 信息的 Tensor 组成的 list，要求每个输入 Tensor 的 LoD 长度必须一致。数据类型为 float32、float64 或 int64。
        - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。

返回
:::::::::
Tensor，输出连接后的 Tensor，数据类型和输入一致。

代码示例
:::::::::

COPY-FROM: paddle.static.nn.sequence_concat
