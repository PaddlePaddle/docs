.. _cn_api_fluid_layers_shard_index:

shard_index
-------------------------------

.. py:function:: paddle.shard_index(input, index_num, nshards, shard_id, ignore_value=-1)


该函数根据分片（shard）的偏移量重新计算分片的索引。索引长度被均分为N个分片，如果输入索引所在的分片跟分片ID对应，则该索引以分片的偏移量为界重新计算，否则更新为默认值（ignore_value）。具体计算为：

::

    shard_size = (index_num + nshards - 1) // nshards
    如果 shard_id == input // shard_size 则 output = input % shard_size  
    否则 output = ignore_value
	
注意：若索引长度不能被分片数整除，则最后一个分片长度不足shard_size。

参数：
    - input (Tensor）-  输入的索引，最后一维的维度值为1，数据类型为int64。
    - index_num (int) - 定义索引长度的整型值。
    - nshards (int) - 分片数量。
    - shard_id (int) - 当前分片ID。
    - ignore_value (int) - 超出分片索引范围的默认值。

返回：更新后的索引值Tensor

**代码示例：**

.. code-block:: python

    import paddle
    label = paddle.to_tensor([[16], [1]], dtype="int64")
    shard_label = paddle.shard_index(input=label,
                                     index_num=20,
                                     nshards=2,
                                     shard_id=0)
    print(shard_label)
    # [[-1], [1]]
