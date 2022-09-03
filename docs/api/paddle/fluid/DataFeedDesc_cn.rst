.. _cn_api_fluid_DataFeedDesc:

DataFeedDesc
-------------------------------


.. py:class:: paddle.fluid.DataFeedDesc(proto_file)




描述训练数据的格式。输入是一个文件路径名，其内容是protobuf message。

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

用户需要了解DataFeedDesc中每个字段的含义，以便自定义字段的值。例如：

.. code-block:: python

    import paddle.fluid as fluid
    data_feed = fluid.DataFeedDesc('data.proto')
    data_feed.set_batch_size(128)
    data_feed.set_dense_slots('words')  # 名为'words'的slot将被设置为密集的
    data_feed.set_use_slots('words')    # 名为'words'的slot将被用于训练

    # 最后，可以打印变量详细信息便于排查错误
    print(data_feed.desc())


参数
::::::::::::

  - **proto_file** (string)：包含数据描述的protobuf message的磁盘文件


方法
::::::::::::
set_batch_size(batch_size)
'''''''''

该接口用于设置DataFeedDesc中的 :code:`batch_size`。可以在训练期间调用修改 :code:`batch_size` 。

**代码示例**

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

**参数**

  - **batch_size** (int) - 新的批尺寸。

**返回**
无

set_dense_slots(dense_slots_name)
'''''''''

将 :code:`dense_slots_name` 指定的slots设置为密集的slot。**注意：默认情况下，所有slots都是稀疏的。**

密集slot的特征将被输入一个Tensor，而稀疏slot的特征将被输入一个LoDTensor。

**代码示例**

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

**参数**

  - **dense_slots_name** (list(str)) - slot名称的列表，这些slot将被设置为密集的。

**返回**
无

set_use_slots(use_slots_name)
'''''''''


设置一个特定的slot是否用于训练。一个数据集包含了很多特征，通过这个函数可以选择哪些特征将用于指定的模型。

**参数**

  - **use_slots_name** (list)：将在训练中使用的slot名列表，类型为list，其中每个元素为一个字符串

**代码示例**

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

  默认值是不使用所有slot


desc()
'''''''''

返回此DataFeedDesc的protobuf message

**返回**
一个protobuf message字符串

**代码示例**

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






