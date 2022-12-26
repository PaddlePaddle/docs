.. _cn_api_fluid_layers_edit_distance:


edit_distance
-------------------------------

.. py:function:: paddle.fluid.layers.edit_distance(input,label,normalized=True,ignored_tokens=None, input_length=None, label_length=None）




该 OP 计算一批给定字符串及其参照字符串间的编辑距离。编辑距离也称 Levenshtein 距离，通过计算从一个字符串变成另一个字符串所需的最少操作步骤来衡量两个字符串的相异度。这里的操作包括插入、删除和替换。

比如给定假设字符串 A=“kitten”和参照字符串 B=“sitting”，从 A 变换成 B 编辑距离为 3，至少需要两次替换和一次插入：

“kitten”->“sitten”->“sittn”->“sitting”

输入为 LoDTensor/Tensor，包含假设字符串（带有表示批尺寸的总数）和分离信息（具体为 LoD 信息或者 ``input_length`` ）。并且批尺寸大小的参照字符串和输入 LoDTensor 的顺序保持一致。

输出包含批尺寸大小的结果，代表一对字符串中每个字符串的编辑距离。如果 Attr(normalized)为真，编辑距离则处以参照字符串的长度。

参数
::::::::::::

    - **input** (Variable) - 假设字符串的索引，rank 为 2 的 Tensor，数据类型为 int64。
    - **label** (Variable) - 参照字符串的索引，rank 为 2 的 Tensor，数据类型为 int64。
    - **normalized** (bool)-表示是否用参照字符串的长度进行归一化，默认值为 True。
    - **ignored_tokens** (list<int>)-计算编辑距离前需要移除的 token，默认值为 None。
    - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。

返回
::::::::::::
包含有形为[batch_size,1]的编辑距离和形为[ ]的序列数元组。

返回类型
::::::::::::
元组

代码示例
::::::::::::

COPY-FROM: paddle.fluid.layers.edit_distance
