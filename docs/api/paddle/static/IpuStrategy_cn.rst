.. _cn_api_fluid_IpuStrategy:

IpuStrategy
-------------------------------


.. py:class:: paddle.static.IpuStrategy()


``IpuStrategy`` 使用户能更精准地控制 :ref:`cn_api_fluid_IpuCompiledProgram` 中计算图的建造方法。


返回
:::::::::
    IpuStrategy实例

代码示例
::::::::::

COPY-FROM: paddle.static.IpuStrategy

.. py:method:: SetGraphConfig(self, num_ipus, is_training, batch_size, enable_manual_shard, need_avg_shard)

该接口用于向IpuStrategy实例传递IPU构图的Graph配置。

参数
:::::::::
    - **num_ipus** (int，可选)- 指定IPU devices的个数，默认值为1，表示仅用一个IPU。
    - **is_training** (bool，可选)- 声明是训练还是推理，默认值为True，表示使用训练模式。
    - **batch_size** (int，可选)- 当计算图输入的batch_size可变时，指定计算图中输入batch_size，默认值为1，表示如果batch_size可变，将默认置1。
    - **enable_manual_shard** (bool，可选)- 是否使能分割计算图到不同IPU进行运算。仅支持当num_ipus > 1时，enable_manual_shard可以置为True。默认值为False，表示不使能该功能。
    - **need_avg_shard** (bool，可选)- 是否使能自动分割计算图到不同IPU进行运算。仅支持当enable_manual_shard=True时，need_avg_shard可以置为True。默认值为False，表示不使能该功能。

代码示例
:::::::::

COPY-FROM: paddle.static.IpuStrategy.SetGraphConfig

.. py:method:: SetPipeliningConfig(self, enable_pipelining, batches_per_step, accumulationFactor)

该接口用于向IpuStrategy实例传递IPU构图的子图数据流水配置。

参数
:::::::::
    - **enable_pipelining** (bool，可选)- 是否使能子图之间的数据流水。仅支持当enable_manual_shard=True时，enable_pipelining可以置为True。默认值为False，表示不使能该功能。
    - **batches_per_step** (int，可选)- 指定数据流水每次运算多少个batch_size的数据。仅支持当enable_pipelining=True时，batches_per_step可以置 > 1。默认值为1，表示不使能数据流水功能。
    - **accumulationFactor** (int，可选)- 指定累积运算多少个batch_size更新一次权重。默认值为1，表示不使能权重累积更新功能。

代码示例
:::::::::

COPY-FROM: paddle.static.IpuStrategy.SetPipeliningConfig

.. py:method:: SetHalfConfig(self, enable_fp16)

该接口用于向IpuStrategy实例传递IPU构图的半精度运算配置。

参数
:::::::::
    - **enable_fp16** (bool)- 是否使能fp16运算模式并将fp32转换为fp16。默认值为False，表示不使能fp16运算模式。

代码示例
:::::::::

COPY-FROM: paddle.static.IpuStrategy.SetHalfConfig

属性
::::::::::::
.. py:attribute:: num_ipus

返回IpuStrategy实例中的IPU设备个数，类型为 ``Int``

.. py:attribute:: is_training

返回IpuStrategy实例中的计算模式是训练模式或推理模式，类型为 ``Bool``

.. py:attribute:: batch_size

返回IpuStrategy实例中的计算图batch_size,用于固定动态的batch_size，类型为 ``Int``

.. py:attribute:: enable_manual_shard

返回IpuStrategy实例中是否使能图切分功能，类型为 ``Bool``

.. py:attribute:: need_avg_shard

返回IpuStrategy实例中是否使能自动切分功能，类型为 ``Bool``

.. py:attribute:: enable_pipelining

返回IpuStrategy实例中是否使能数据流水功能，类型为 ``Parameter``

.. py:attribute:: batches_per_step

返回IpuStrategy实例中的数据流水模式下每次执行的batch_size个数，类型为 ``Int``

.. py:attribute:: accumulationFactor

返回IpuStrategy实例中的训练模式下权重累积batch个数，类型为 ``Int``

.. py:attribute:: enable_fp16

返回IpuStrategy实例中是否使能float16计算图，类型为 ``Bool``