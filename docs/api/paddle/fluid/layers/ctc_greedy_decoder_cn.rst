.. _cn_api_fluid_layers_ctc_greedy_decoder:

ctc_greedy_decoder
-------------------------------

.. py:function:: paddle.fluid.layers.ctc_greedy_decoder(input, blank, name=None)





该OP用于贪婪策略解码序列，步骤如下：
    1. 获取输入中的每一行的最大值索引，也就是numpy.argmax(input, axis=0)。
    2. 对于step1结果中的每个序列，合并两个空格之间的重复部分并删除所有空格。

该API支持两种输入，LoDTensor和Tensor输入，不同输入的代码样例如下：

**样例**：

::

        # for lod tensor input 
        已知：

        input.data = [[0.6, 0.1, 0.3, 0.1],
                      [0.3, 0.2, 0.4, 0.1],
                      [0.1, 0.5, 0.1, 0.3],
                      [0.5, 0.1, 0.3, 0.1],

                      [0.5, 0.1, 0.3, 0.1],
                      [0.2, 0.2, 0.2, 0.4],
                      [0.2, 0.2, 0.1, 0.5],
                      [0.5, 0.1, 0.3, 0.1]]

        input.lod = [[4, 4]]

        计算过程：

        1. 将argmax的运算结果应用于输入的第一个序列，即 input.data[0:4] 。
           则得出的结果为[[0], [2], [1], [0]]
        2. 合并重复的索引值部分，删除空格，即为0的值。
           则第一个输入序列对应的输出为：[[2], [1]]

        最后

        output.data = [[2],
                       [1],
                       [3]]

        output.lod = [[2, 1]]

        # for tensor input
        input.data = [[[0.6, 0.1, 0.3, 0.1],
                [0.3, 0.2, 0.4, 0.1],
                [0.1, 0.5, 0.1, 0.3],
                [0.5, 0.1, 0.3, 0.1]],

               [[0.5, 0.1, 0.3, 0.1],
                [0.2, 0.2, 0.2, 0.4],
                [0.2, 0.2, 0.1, 0.5],
                [0.5, 0.1, 0.3, 0.1]]]

        input_length.data = [[4], [4]]
        input.shape = [2, 4, 4]

        step1: Apply argmax to first input sequence which is input.data[0:4]. Then we get:
            [[0], [2], [1], [0]], for input.data[4:8] is [[0], [3], [3], [0]], shape is [2,4,1]
        step2: Change the argmax result to use padding mode, then argmax result is
                [[0, 2, 1, 0], [0, 3, 3, 0]], shape is [2, 4], lod is [], input_length is [[4], [4]]
        step3: Apply ctc_align to padding argmax result, padding_value is 0

        Finally:
        output.data = [[2, 1, 0, 0],
                       [3, 0, 0, 0]]
        output_length.data = [[2], [1]]


参数
::::::::::::

        - **input** (Variable) — 变长序列的概率，在输入为LoDTensor情况下，它是具有LoD信息的二维LoDTensor。形状为[Lp，num_classes +1]，其中Lp是所有输入序列的长度之和，num_classes是真实的类数。在输入为Tensor情况下，它是带有填充的3-D张量，其形状为[batch_size，N，num_classes +1]。 （不包括空白标签）。数据类型可以是float32或float64。
        - **blank** (int) — Connectionist Temporal Classification (CTC) loss空白标签索引，其数值属于半开区间[0,num_classes + 1）
        - **name** (str) — (str|None，可选) – 该参数供开发人员打印调试信息时使用，具体用法请参见 :ref:`api_guide_Name`，默认值为None

返回
::::::::::::
对于输入为LoDTensor的情况，返回CTC贪婪解码器的结果，即2-D LoDTensor，形状为[Lp，1]，数据类型为int64。“Lp”是所有输出序列长度的总和。如果结果中的所有序列均为空，则结果LoDTensor将为[-1]，其中LoD为[[]]。

对于输入为Tensor的情况，返回一个元组，(output, output_length)，其中，output是一个形状为 [batch_size, N]，类型为int64的Tensor。output_length是一个形状为[batch_size, 1]，类型为int64的Tensor，表示Tensor输入下，每个输出序列的长度。

返回类型
::::::::::::
 Variable


代码示例
::::::::::::

COPY-FROM: paddle.fluid.layers.ctc_greedy_decoder