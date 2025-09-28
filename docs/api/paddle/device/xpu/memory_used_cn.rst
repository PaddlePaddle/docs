.. _cn_api_paddle_device_xpu_memory_used:

memory_used
-------------------------------

.. py:function:: paddle.device.xpu.memory_used(device=None)

返回给定设备上 XPU 运行时已使用的显存大小。

.. note::
    这个 API 返回的显存使用量由 XPU RUNTIME 管理，因此可能比 Paddle 管理的显存使用量更大。

参数
::::::::

    - **device** (paddle.XPUPlace|int|str，可选) - 设备、设备 ID 或形如 ``xpu:x`` 的设备名称。如果 ``device`` 为 None，则 ``device`` 为当前的设备。默认值为 None。


返回
::::::::

一个整数，表示指定设备上 XPU 运行时已使用的显存量，以字节为单位。

代码示例
::::::::

COPY-FROM: paddle.device.xpu.memory_used
