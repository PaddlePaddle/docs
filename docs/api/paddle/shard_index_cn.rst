.. _cn_api_fluid_layers_shard_index:

shard_index
-------------------------------

.. py:function:: paddle.shard_index(input, index_num, nshards, shard_id, ignore_value=-1)

根据当前 shard 重新设置输入参数\ `input`\ 的值。输入\ `input`\ 中的值需要为非负整型；参数\ `index_num`\ 为用户设置的大于\ `input`\ 最大值的整型值。因此，\ `input`\ 中的值属于区间[0, index_num)，且每个值可以被看作到区间起始的偏移量。区间可以被进一步划分为多个切片。具体地讲，我们首先根据下面的公式计算每个切片的大小：\ `shard_size`\，表示每个切片可以表示的整数的数量。因此，对于第\ `i`\ 个切片，其表示的区间为[i*shard_size, (i+1)*shard_size)。

::

    shard_size = (index_num + nshards - 1) // nshards

对于输入\ `input`\ 中的每个值\ `v`\，我们根据下面的公式设置它新的值：

::

    v = v - shard_id * shard_size if shard_id * shard_size <= v < (shard_id+1) * shard_size else ignore_value

参数
::::::::::::

    - **input** (Tensor) - 输入 tensor，最后一维的维度值为 1，数据类型为 int64 或 int32。
    - **index_num** (int) - 用户设置的大于 :attr:`input` 最大值的整型值。
    - **nshards** (int) - 分片数量。
    - **shard_id** (int) - 当前分片 ID。
    - **ignore_value** (int，可选) - 超出分片索引范围的整数值。默认值为 -1。

返回
::::::::::::
Tensor

代码示例
::::::::::::

COPY-FROM: paddle.shard_index
