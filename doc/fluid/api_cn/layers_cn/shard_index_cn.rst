.. _cn_api_fluid_layers_shard_index:

shard_index
-------------------------------

.. py:function:: paddle.fluid.layers.shard_index(input, index_num, nshards, shard_id, ignore_value=-1)

该层为输入创建碎片化索引，通常在模型和数据并行混合训练时使用，索引数据（通常是标签）应该在每一个trainer里面被计算，通过
::

    assert index_num % nshards == 0

    shard_size = index_num / nshards

    如果 x / shard_size == shard_id

    y = x % shard_size  

    否则

    y = ignore_value

我们使用分布式 ``one-hot`` 表示来展示该层如何使用， 分布式的 ``one-hot`` 表示被分割为多个碎片, 碎片索引里不为1的都使用0来填充。为了在每一个trainer里面创建碎片化的表示，原始的索引应该先进行计算(i.e. sharded)。我们来看个例子：

.. code-block:: text

    X 是一个整形张量
    X.shape = [4, 1]
    X.data = [[1], [6], [12], [19]]

    假设 index_num = 20 并且 nshards = 2, 我们可以得到 shard_size = 10

    如果 shard_id == 0, 我们得到输出:
    Out.shape = [4, 1]
    Out.data = [[1], [6], [-1], [-1]]
    如果 shard_id == 1, 我们得到输出:
    Out.shape = [4, 1]
    Out.data = [[-1], [-1], [2], [9]]

    上面的例子中默认 ignore_value = -1

参数：
        - **input** (Variable）-  输入的索引，最后的维度应该为1
        - **index_num** (scalar) - 定义索引长度的整形参数
        - **nshards** (scalar) - shards数量
        - **shard_id** (scalar) - 当前碎片的索引
        - **ignore_value** (scalar) - 超出碎片索引范围的整型值

返回： 输入的碎片索引

返回类型：    Variable

**代码示例：**

.. code-block:: python

    import paddle.fluid as fluid
    label = fluid.layers.data(name="label", shape=[1], dtype="int64")
    shard_label = fluid.layers.shard_index(input=label,
                                       index_num=20,
                                       nshards=2,
                                       shard_id=0)





