.. _cn_api_fluid_IpuGraphConfig:

IpuGraphConfig
-------------------------------


.. py::function:: paddle.static.IpuGraphConfig

:api_attr: 声明式编程模式（静态图)

通过``IpuGraphConfig`` 向IpuConfig实例中传递IPU构图的Graph配置。


参数：
    - **ipu_config** (IpuConfig)- IpuConfig实例，可通过:ref:`cn_api_fluid_IpuConfig`获取。
    - **num_ipus** (int)- 指定IPU devices的个数，默认值为1。
    - **is_training** (bool)- 声明是训练还是推理，默认值为True。
    - **batch_size** (int)- 当计算图输入的batch_size可变时，指定计算图中输入batch_size，默认值为1。
    - **enable_manual_shard** (bool)- 是否使能分割计算图到不同IPU进行运算。仅支持当num_ipus > 1时，enable_manual_shard可以置为True。默认值为False。
    - **need_avg_shard** (bool)- 是否使能自动分割计算图到不同IPU进行运算。仅支持当enable_manual_shard=True时，need_avg_shard可以置为True。默认值为False。

代码示例
::::::::::

.. code-block:: python
	
    # required: ipu
    
    import paddle
    import paddle.static as static
            
    paddle.enable_static()
    ipu_config = static.IpuConfig()
    static.IpuGraphConfig(ipu_config,
                          num_ipus=1, 
                          is_training=True,
                          batch_size=1,
                          enable_manual_shard=False,
                          need_avg_shard=False)