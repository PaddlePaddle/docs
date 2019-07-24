.. _cn_api_fluid_DataFeedDesc:

DataFeedDesc
-------------------------------

.. py:class:: paddle.fluid.DataFeedDesc(proto_file)

数据描述符，描述输入训练数据格式。

这个类目前只用于AsyncExecutor(有关类AsyncExecutor的简要介绍，请参阅注释)

DataFeedDesc应由来自磁盘的有效protobuf消息初始化。

可以参考 :code:`paddle/fluid/framework/data_feed.proto` 查看我们如何定义message

一段典型的message可能是这样的：

.. code-block:: python

    import paddle.fluid as fluid
    f = open("data.proto", "w")
    print >> f, 'name: "MultiSlotDataFeed"'
    print >> f, 'batch_size: 2'
    print >> f, 'multi_slot_desc {'
    print >> f, '    slots {'
    print >> f, '         name: "words"'
    print >> f, '         type: "uint64"'
    print >> f, '         is_dense: false'
    print >> f, '         is_used: true'
    print >> f, '     }'
    print >> f, '     slots {'
    print >> f, '         name: "label"'
    print >> f, '         type: "uint64"'
    print >> f, '         is_dense: false'
    print >> f, '         is_used: true'
    print >> f, '    }'
    print >> f, '}'
    f.close()
    data_feed = fluid.DataFeedDesc('data.proto')

但是，用户通常不应该关心消息格式;相反，我们鼓励他们在将原始日志文件转换为AsyncExecutor可以接受的训练文件的过程中，使用 :code:`Data Generator` 生成有效数据描述。

DataFeedDesc也可以在运行时更改。一旦你熟悉了每个字段的含义，您可以修改它以更好地满足您的需要。例如:

.. code-block:: python

    import paddle.fluid as fluid
    data_feed = fluid.DataFeedDesc('data.proto')
    data_feed.set_batch_size(128)
    data_feed.set_dense_slots('wd')  # 名为'wd'的slot将被设置为密集的
    data_feed.set_use_slots('wd')    # 名为'wd'的slot将被用于训练

    # 最后，可以打印变量详细信息便于排出错误

    print(data_feed.desc())


参数：
  - **proto_file** (string) - 包含数据feed中描述的磁盘文件


.. py:method:: set_batch_size(batch_size)

设置batch size，训练期间有效


参数：
  - batch_size：batch size

**代码示例：**

.. code-block:: python

    import paddle.fluid as fluid
    f = open("data.proto", "w")
    print >> f, 'name: "MultiSlotDataFeed"'
    print >> f, 'batch_size: 2'
    print >> f, 'multi_slot_desc {'
    print >> f, '    slots {'
    print >> f, '         name: "words"'
    print >> f, '         type: "uint64"'
    print >> f, '         is_dense: false'
    print >> f, '         is_used: true'
    print >> f, '     }'
    print >> f, '     slots {'
    print >> f, '         name: "label"'
    print >> f, '         type: "uint64"'
    print >> f, '         is_dense: false'
    print >> f, '         is_used: true'
    print >> f, '    }'
    print >> f, '}'
    f.close()
    data_feed = fluid.DataFeedDesc('data.proto')
    data_feed.set_batch_size(128)

.. py:method:: set_dense_slots(dense_slots_name)

指定slot经过设置后将变成密集的slot，仅在训练期间有效。

密集slot的特征将被输入一个Tensor，而稀疏slot的特征将被输入一个lodTensor


参数：
  - **dense_slots_name** : slot名称的列表，这些slot将被设置为密集的

**代码示例：**

.. code-block:: python

    import paddle.fluid as fluid
    f = open("data.proto", "w")
    print >> f, 'name: "MultiSlotDataFeed"'
    print >> f, 'batch_size: 2'
    print >> f, 'multi_slot_desc {'
    print >> f, '    slots {'
    print >> f, '         name: "words"'
    print >> f, '         type: "uint64"'
    print >> f, '         is_dense: false'
    print >> f, '         is_used: true'
    print >> f, '     }'
    print >> f, '     slots {'
    print >> f, '         name: "label"'
    print >> f, '         type: "uint64"'
    print >> f, '         is_dense: false'
    print >> f, '         is_used: true'
    print >> f, '    }'
    print >> f, '}'
    f.close()
    data_feed = fluid.DataFeedDesc('data.proto')
    data_feed.set_dense_slots(['words'])

.. note::

  默认情况下，所有slot都是稀疏的

.. py:method:: set_use_slots(use_slots_name)


设置一个特定的slot是否用于训练。一个数据集包含了很多特征，通过这个函数可以选择哪些特征将用于指定的模型。

参数：
  - **use_slots_name** :将在训练中使用的slot名列表

**代码示例：**

.. code-block:: python
    
    import paddle.fluid as fluid
    f = open("data.proto", "w")
    print >> f, 'name: "MultiSlotDataFeed"'
    print >> f, 'batch_size: 2'
    print >> f, 'multi_slot_desc {'
    print >> f, '    slots {'
    print >> f, '         name: "words"'
    print >> f, '         type: "uint64"'
    print >> f, '         is_dense: false'
    print >> f, '         is_used: true'
    print >> f, '     }'
    print >> f, '     slots {'
    print >> f, '         name: "label"'
    print >> f, '         type: "uint64"'
    print >> f, '         is_dense: false'
    print >> f, '         is_used: true'
    print >> f, '    }'
    print >> f, '}'
    f.close()
    data_feed = fluid.DataFeedDesc('data.proto')
    data_feed.set_use_slots(['words'])

.. note::

  默认值不用于所有slot


.. py:method:: desc()

返回此DataFeedDesc的protobuf信息

返回：一个message字符串

**代码示例：**

.. code-block:: python
    
    import paddle.fluid as fluid
    f = open("data.proto", "w")
    print >> f, 'name: "MultiSlotDataFeed"'
    print >> f, 'batch_size: 2'
    print >> f, 'multi_slot_desc {'
    print >> f, '    slots {'
    print >> f, '         name: "words"'
    print >> f, '         type: "uint64"'
    print >> f, '         is_dense: false'
    print >> f, '         is_used: true'
    print >> f, '     }'
    print >> f, '     slots {'
    print >> f, '         name: "label"'
    print >> f, '         type: "uint64"'
    print >> f, '         is_dense: false'
    print >> f, '         is_used: true'
    print >> f, '    }'
    print >> f, '}'
    f.close()
    data_feed = fluid.DataFeedDesc('data.proto')
    print(data_feed.desc())






