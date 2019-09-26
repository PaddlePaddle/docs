.. _cn_api_fluid_layers_shard_index:

shard_index
-------------------------------

.. py:function:: paddle.fluid.layers.shard_index(input, index_num, nshards, shard_id, ignore_value=-1)

该函数对输入的索引根据分片（shard）的偏移量重新计算。
索引长度被均分为N个分片，如果输入索引所在的分片跟分片ID对应，则该索引以分片的偏移量为界重新计算，否则更新为默认值（ignore_value）。具体计算为：
::

    每个分片的长度为
    shard_size = (index_num + nshards - 1) // nshards

    如果 shard_id == input // shard_size
    则 output = input % shard_size  
    否则 output = ignore_value
	
注意：若索引长度不能被分片数整除，则最后一个分片长度不足shard_size。

示例：
::

    输入：
    input.shape = [4, 1]
    input.data = [[1], [6], [12], [19]]
    index_num = 20
    nshards = 2
    ignore_value=-1

    如果 shard_id == 0, 输出:
    output.shape = [4, 1]
    output.data = [[1], [6], [-1], [-1]]

    如果 shard_id == 1, 输出:
    output.shape = [4, 1]
    output.data = [[-1], [-1], [2], [9]]

参数：
    - **input** (Variable）-  输入的索引
    - **index_num** (scalar) - 索引长度
    - **nshards** (scalar) - 分片数量
    - **shard_id** (scalar) - 当前分片ID
    - **ignore_value** (scalar) - 超出分片索引范围的默认值

返回：更新后的索引值

返回类型：Variable

**代码示例：**

.. code-block:: python

    import paddle.fluid as fluid
    label = fluid.layers.data(name="label", shape=[1], dtype="int64")
    shard_label = fluid.layers.shard_index(input=label,
                                           index_num=20,
                                           nshards=2,
                                           shard_id=0)
