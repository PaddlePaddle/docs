.. _cn_api_paddle_device_memory_allocated:

memory_allocated
---------------

.. py:function:: paddle.device.memory_allocated(device=None)

返回给定设备上当前分配给 Tensor 的内存大小。

.. note::
    Paddle 中分配给 Tensor 的内存块大小会进行 256 字节对齐，因此可能大于 Tensor 实际需要的内存大小。例如，一个 shape 为[1]的 float32 类型 Tensor 会占用 256 字节的内存，即使存储一个 float32 类型数据实际只需要 4 字节。

参数
::::::::::::

    - **device** (paddle.CUDAPlace|paddle.CustomPlace|paddle.XPUPlace|str|int，可选) - 设备、设备 ID 或形如 ``gpu:x``、``xpu:x`` 或自定义设备名称的设备字符串。如果为 None，则返回当前设备的统计信息。默认值为 None。

返回
::::::::::::

    int，给定设备上当前分配给 Tensor 的内存大小，以字节为单位。

代码示例
::::::::::::
.. code-block:: pycon

    >>> import paddle
    >>> paddle.device.memory_allocated('npu:0')
    >>> paddle.device.memory_allocated('npu')
    >>> paddle.device.memory_allocated(0)
    >>> paddle.device.memory_allocated(paddle.CustomPlace('npu', 0))
