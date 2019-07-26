.. _cn_api_fluid_layers_shuffle:

shuffle
-------------------------------

.. py:function:: paddle.fluid.layers.shuffle(reader, buffer_size)

创建一个特殊的数据读取器，它的输出数据会被重洗(shuffle)。由原始读取器创建的迭代器得到的输出将会被暂存到shuffle缓存区，其后
会对其进行重洗运算。shuffle缓存区的大小由参数 ``buffer_size`` 决定。

参数:
    - **reader** (callable) – 输出会被shuffle的原始reader
    - **buffer_size** (int) – 进行shuffle的buffer的大小

返回:其输出会被shuffle的一个reader（读取器）

返回类型:callable

**代码示例**：

.. code-block:: python

    import paddle.fluid as fluid
    raw_reader = fluid.layers.io.open_files(filenames=['./data1.recordio',
                                                   './data2.recordio'],
                                            shapes=[(3,224,224), (1,)],
                                            lod_levels=[0, 0],
                                            dtypes=['float32', 'int64'],
                                            thread_num=2,
                                            buffer_size=2)
    batch_reader = fluid.layers.batch(reader=raw_reader, batch_size=5)
    shuffle_reader = fluid.layers.shuffle(reader=batch_reader, buffer_size=5000)








