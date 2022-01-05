.. _cn_api_fluid_IpuPipeliningConfig:

IpuPipeliningConfig
-------------------------------


.. py::function:: paddle.static.IpuPipeliningConfig(ipu_config, enable_pipelining=False, batches_per_step=1, accumulationFactor=1)

通过 ``IpuPipeliningConfig`` 向IpuConfig实例中传递IPU构图的子图数据流水配置。


参数：
    - **ipu_config** (IpuConfig)- IpuConfig实例，可通过 :ref:`cn_api_fluid_IpuConfig` 获取。
    - **enable_pipelining** (bool，可选)- 是否使能子图之间的数据流水。仅支持当enable_manual_shard=True时，enable_pipelining可以置为True。默认值为False，表示不使能该功能。
    - **batches_per_step** (int，可选)- 指定数据流水每次运算多少个batch_size的数据。仅支持当enable_pipelining=True时，batches_per_step可以置 > 1。默认值为1，表示不使能数据流水功能。
    - **accumulationFactor** (int，可选)- 指定累积运算多少个batch_size更新一次权重。默认值为1，表示不使能权重累积更新功能。

代码示例
::::::::::

.. code-block:: python
	
    # required: ipu
    
    import paddle
    import paddle.static as static
            
    paddle.enable_static()
    ipu_config = static.IpuConfig()
    static.IpuPipeliningConfig(ipu_config,
                               enable_pipelining=False,
                               batches_per_step=1,
                               accumulationFactor=1)