.. _cn_api_fluid_IpuConfig:

IpuConfig
-------------------------------


.. py::function:: paddle.static.IpuConfig()

:api_attr: 声明式编程模式（静态图)

``IpuConfig`` 使用户能更精准地控制 :ref:`cn_api_fluid_IpuCompiledProgram` 中计算图的建造方法，调用``IpuConfig``成员来获取对象。


返回：IpuConfig实例。接收:ref:`cn_api_fluid_IpuGraphConfig`，:ref:`cn_api_fluid_IpuPipeliningConfig`，:ref:`cn_api_fluid_IpuHalfConfig`等接口传入的IPU构图配置信息，以及作为:ref:`cn_api_fluid_IpuCompiledProgram`的参数传递构图配置。
    

代码示例
::::::::::

.. code-block:: python
	
    # required: ipu
    
    import paddle
    import paddle.static as static
            
    paddle.enable_static()
    ipu_config = static.IpuConfig()
