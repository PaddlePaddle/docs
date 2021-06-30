.. _cn_api_get_device:

get_device
-------------------------------

.. py:function:: paddle.get_device()


该功能返回当前程序运行的全局设备，返回的是一个类似于 ``cpu``、 ``gpu:x``、 ``xpu:x`` 或者 ``npu:0`` 字符串，如果没有设置全局设备，当cuda可用的时候返回 ``gpu:0`` ，当cuda不可用的时候返回 ``cpu`` 。

返回：返回当前程序运行的全局设备。

**代码示例**

.. code-block:: python
        
    import paddle
    
    device = paddle.get_device()
