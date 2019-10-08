.. _cn_api_fluid_layers_ctc_greedy_decoder:

ctc_greedy_decoder
-------------------------------

.. py:function:: paddle.fluid.layers.ctc_greedy_decoder(input, blank, name=None)

**注意：该OP的输入input必须是2维LoDTensor, lod_level为1** 

该OP用于贪婪策略解码序列，步骤如下:
    1. 获取输入中的每一行的最大值索引，也就是numpy.argmax(input, axis=0)。
    2. 对于step1结果中的每个序列，合并两个空格之间的重复部分并删除所有空格。


**样例**：

::

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


参数:
        - **input** (Variable) — 变长序列的概率，2维LoDTensor, lod_level为1。它的形状是[Lp, num_classes + 1]，其中Lp是所有输入序列长度的和，num_classes是类别数目(不包括空白标签)。数据类型是float32或者float64
        - **blank** (int) — Connectionist Temporal Classification (CTC) loss空白标签索引,  其数值属于半开区间[0,num_classes + 1）
        - **name** (str) — (str|None，可选) – 该参数供开发人员打印调试信息时使用，具体用法请参见 :ref:`api_guide_Name` ，默认值为None

返回： CTC贪婪解码结果是一个形为(Lp,1)的2维LoDTensor，lod_level为1，其中Lp是所有输出序列的长度之和。如果结果中的所有序列都为空，则输出LoDTensor为[-1]，其lod信息为空。

返回类型： Variable


**代码示例**

..  code-block:: python

    import paddle.fluid as fluid
    x = fluid.layers.data(name='x', shape=[8], dtype='float32')
    cost = fluid.layers.ctc_greedy_decoder(input=x, blank=0)





