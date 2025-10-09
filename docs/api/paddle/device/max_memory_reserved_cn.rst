.. _cn_api_paddle_device_max_memory_reserved:

max_memory_reserved
-------------------

.. py:function:: paddle.device.max_memory_reserved(device=None)

返回给定设备上由内存分配器管理的内存峰值统计。

参数
::::::::::::

    - **device** (paddle.CUDAPlace|paddle.CustomPlace|paddle.XPUPlace|str|int，可选) - 设备、设备 ID 或形如 ``gpu:x``、``xpu:x`` 或自定义设备名称的设备字符串。如果为 None，则返回当前设备的统计信息。默认值为 None。

返回
::::::::::::

    int，给定设备上由内存分配器管理的内存峰值统计，以字节为单位。

代码示例
::::::::::::
.. code-block:: python

    >>> import paddle
    >>> paddle.device.max_memory_reserved('npu:0')
    >>> paddle.device.max_memory_reserved('npu')
    >>> paddle.device.max_memory_reserved(0)
    >>> paddle.device.max_memory_reserved(Paddle.CustomPlace('npu',0))
