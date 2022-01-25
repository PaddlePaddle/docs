.. _cn_api_get_available_device:

get_available_device
-------------------------------

.. py:function:: paddle.device.get_available_device()


该功能获得所有可用的设备。

返回
:::::::::
返回列表包含所有可用的设备。

代码示例
:::::::::

.. code-block:: python
        
    import paddle
    
    device = paddle.device.get_available_device()
    # ['cpu', 'gpu:0', 'gpu:1', 'FakeCPU', 'FakeGPU:0', 'FakeGPU:1']