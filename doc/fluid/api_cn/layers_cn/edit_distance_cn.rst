.. _cn_api_fluid_layers_edit_distance:


edit_distance
-------------------------------

.. py:function:: paddle.fluid.layers.edit_distance(input,label,normalized=True,ignored_tokens=None, input_length=None, label_length=None）

编辑距离算子

计算一批给定字符串及其参照字符串间的编辑距离。编辑距离也称Levenshtein距离，通过计算从一个字符串变成另一个字符串所需的最少操作步骤来衡量两个字符串的相异度。这里的操作包括插入、删除和替换。

比如给定假设字符串A=“kitten”和参照字符串B=“sitting”，从A变换成B编辑距离为3，至少需要两次替换和一次插入：

“kitten”->“sitten”->“sittn”->“sitting”

输入为LoDTensor/Tensor,包含假设字符串（带有表示批尺寸的总数）和分离信息（具体为LoD信息或者 ``input_length`` ）。并且批尺寸大小的参照字符串和输入LoDTensor的顺序保持一致。

输出包含批尺寸大小的结果，代表一对字符串中每个字符串的编辑距离。如果Attr(normalized)为真，编辑距离则处以参照字符串的长度。

参数：
    - **input** (Variable)-假设字符串的索引，为两列并且类型为int64
    - **label** (Variable)-参照字符串的索引，为两列并且类型为int64
    - **normalized** (bool,默认为True)-表示是否用参照字符串的长度进行归一化
    - **ignored_tokens** (list<int>,默认为None)-计算编辑距离前需要移除的token
    - **name** (str)-该层名称，可选

返回：形为[batch_size,1]的编辑距离。
sequence_num(Variable):形为[ ]的序列数

**代码示例**

.. code-block:: python

    import paddle.fluid as fluid

    # 使用 LoDTensor
    x_lod = fluid.layers.data(name='x_lod', shape=[1], dtype='int64', lod_level=1)
    y_lod = fluid.layers.data(name='y_lod', shape=[1], dtype='int64', lod_level=1)
    distance_lod, seq_num_lod = fluid.layers.edit_distance(input=x_lod, label=y_lod)

    # 使用 Tensor
    x_seq_len = 5
    y_seq_len = 6
    x_pad = fluid.layers.data(name='x_pad', shape=[x_seq_len], dtype='int64')
    y_pad = fluid.layers.data(name='y_pad', shape=[y_seq_len], dtype='int64')
    x_len = fluid.layers.data(name='x_len', shape=[], dtype='int64')
    y_len = fluid.layers.data(name='y_len', shape=[], dtype='int64')
    distance_pad, seq_num_pad = fluid.layers.edit_distance(
                    input=x_pad, label=y_pad, input_length=x_len, label_length=y_len)