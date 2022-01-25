.. _cn_api_get_available_custom_device:

get_available_custom_device
-------------------------------

.. py:function:: paddle.device.get_available_custom_device()


该功能获得所有可用的自定义设备。

返回
:::::::::
返回列表包含所有可用的自定义设备。

代码示例
:::::::::

.. code-block:: python
        
    import paddle
    
    device = paddle.device.get_available_custom_device()
    # ['FakeCPU', 'FakeGPU:0', 'FakeGPU:1']