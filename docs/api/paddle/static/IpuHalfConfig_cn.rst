.. _cn_api_fluid_IpuHalfConfig:

IpuHalfConfig
-------------------------------


.. py::function:: paddle.static.IpuHalfConfig

:api_attr: 声明式编程模式（静态图)

通过``IpuHalfConfig`` 向IpuConfig实例中传递IPU构图的半精度运算配置。


参数：
    - **ipu_config** (IpuConfig)- IpuConfig实例，可通过 :ref:`cn_api_fluid_IpuConfig` 获取。
    - **enable_fp16** (bool)- 是否使能fp16运算模式并将fp32转换为fp16。默认值为False，表示不使能fp16运算模式。

代码示例
::::::::::

.. code-block:: python
	
    # required: ipu
    
    import paddle
    import paddle.static as static
            
    paddle.enable_static()
    ipu_config = static.IpuConfig()
    static.IpuHalfConfig(ipu_config, enable_fp16=False)