.. _cn_api_paddle_device_memory_reserved:

memory_reserved
---------------

.. py:function:: paddle.device.memory_reserved(device=None)

返回给定设备上当前由内存分配器管理的内存大小。

参数
::::::::::::

    - **device** (paddle.CUDAPlace|paddle.CustomPlace|paddle.XPUPlace|str|int，可选) - 设备、设备 ID 或形如 ``gpu:x``、``xpu:x`` 或自定义设备名称的设备字符串。如果为 None，则返回当前设备的统计信息。默认值为 None。

返回
::::::::::::

    int，给定设备上当前由内存分配器管理的内存大小，以字节为单位。

代码示例
::::::::::::
.. code-block:: pycon

    >>> import paddle
    >>> paddle.device.memory_reserved('npu:0')
    >>> paddle.device.memory_reserved('npu')
    >>> paddle.device.memory_reserved(0)
    >>> paddle.device.memory_reserved(paddle.CustomPlace('npu', 0))
