######
fluid
######
.. _cn_api_fluid_AsyncExecutor:

AsyncExecutor
>>>>>>>>>>>>>>>

.. py:class:: paddle.fluid.AsyncExecutor(place=None)

Python中的异步执行器。AsyncExecutor利用多核处理器和数据排队的强大功能，使数据读取和融合解耦，每个线程并行运行。

AsyncExecutor不是在python端读取数据，而是接受一个训练文件列表，该列表将在c++中检索，然后训练输入将被读取、解析并在c++代码中提供给训练网络。

AsyncExecutor正在积极开发，API可能在不久的将来会发生变化。

参数：
	- **place** (fluid.CPUPlace|None) - 指示 executor 将在哪个设备上运行。目前仅支持CPU

**代码示例：**

.. code-block:: python

    data_feed = fluid.DataFeedDesc('data.proto')
    startup_program = fluid.default_startup_program()
    main_program = fluid.default_main_program()
    filelist = ["train_data/part-%d" % i for i in range(100)]
    thread_num = len(filelist) / 4
    place = fluid.CPUPlace()
    async_executor = fluid.AsyncExecutor(place)
    async_executor.run_startup_program(startup_program)
    epoch = 10
    for i in range(epoch):
        async_executor.run(main_program,
                           data_feed,
                           filelist,
                           thread_num,
                           [acc],
                           debug=False)

.. note::

	对于并行gpu调试复杂网络，您可以在executor上测试。他们有完全相同的参数，并可以得到相同的结果。

	目前仅支持CPU

.. _cn_api_fluid_DataFeedDesc:

DataFeedDesc
>>>>>>>>>>>>>>

.. py:class:: paddle.fluid.DataFeedDesc(proto_file)

数据描述符，描述输入训练数据格式。

这个类目前只用于AsyncExecutor(有关类AsyncExecutor的简要介绍，请参阅注释)

DataFeedDesc应由来自磁盘的有效protobuf消息初始化:

.. code-block:: python

	data_feed = fluid.DataFeedDesc('data.proto')

可以参考 :code:`paddle/fluid/framework/data_feed.proto` 查看我们如何定义message

一段典型的message可能是这样的：

.. code-block:: text

    name: "MultiSlotDataFeed"
    batch_size: 2
    multi_slot_desc {
        slots {
            name: "words"
            type: "uint64"
            is_dense: false
            is_used: true
        }
        slots {
            name: "label"
            type: "uint64"
            is_dense: false
            is_used: true
        }
    }

但是，用户通常不应该关心消息格式;相反，我们鼓励他们在将原始日志文件转换为AsyncExecutor可以接受的训练文件的过程中，使用 :code:`Data Generator` 生成有效数据描述。

DataFeedDesc也可以在运行时更改。一旦你熟悉了每个字段的含义，您可以修改它以更好地满足您的需要。例如:

.. code-block:: text

    data_feed.set_batch_size(128)
    data_feed.set_dense_slots('wd')  # The slot named 'wd' will be dense
    data_feed.set_use_slots('wd')    # The slot named 'wd' will be used
    
    #Finally, the content can be dumped out for debugging purpose:
    
    print(data_feed.desc())

参数：
	- **proto_file** (string) - 包含数据feed中描述的磁盘文件


.. py:method:: set_batch_size(self, batch_size)

设置batch size，训练期间有效


参数：
	- batch_size：batch size

**代码示例：**

.. code-block:: python
	
	data_feed = fluid.DataFeedDesc('data.proto')
	data_feed.set_batch_size(128)

.. py:method:: set_dense_slots(self, dense_slots_name)

指定slot经过设置后将变成密集的slot，仅在训练期间有效。

密集slot的特征将被输入一个Tensor，而稀疏slot的特征将被输入一个lodTensor


参数：
	- **dense_slots_name** : slot名称的列表，这些slot将被设置为密集的

**代码示例：**

.. code-block:: python
	
	data_feed = fluid.DataFeedDesc('data.proto')
	data_feed.set_dense_slots(['words'])

.. note:: 

	默认情况下，所有slot都是稀疏的

.. py:method:: set_use_slots(self, use_slots_name)


设置一个特定的slot是否用于训练。一个数据集包含了很多特征，通过这个函数可以选择哪些特征将用于指定的模型。

参数：
	- **use_slots_name** :将在训练中使用的slot名列表

**代码示例：**

.. code-block:: python

	data_feed = fluid.DataFeedDesc('data.proto')
	data_feed.set_use_slots(['words'])

.. note::
	
	默认值不用于所有slot


.. py:method:: desc(self)

返回此DataFeedDesc的protobuf信息

返回：一个message字符串

**代码示例：**

.. code-block:: python

	data_feed = fluid.DataFeedDesc('data.proto')
	print(data_feed.desc())