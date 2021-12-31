.. _cn_api_fluid_IpuConfig:

IpuConfig
-------------------------------


.. py::function:: paddle.static.IpuConfig()


``IpuConfig`` 使用户能更精准地控制 :ref:`cn_api_fluid_IpuCompiledProgram` 中计算图的建造方法。

返回
::::::::::
    IpuConfig实例。
    

代码示例
::::::::::

.. code-block:: python
	
    # required: ipu
    
    import paddle
    import paddle.static as static
            
    paddle.enable_static()
    ipu_config = static.IpuConfig()
