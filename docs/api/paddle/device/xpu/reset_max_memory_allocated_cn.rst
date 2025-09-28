.. _cn_api_paddle_device_xpu_reset_max_memory_allocated:

reset_max_memory_allocated
-------------------------------

.. py:function:: paddle.device.xpu.reset_max_memory_allocated(device=None)

重置给定设备上分配给 Tensor 的显存峰值统计。

参数
::::::::

    - **device** (paddle.XPUPlace|int|str，可选) - 设备、设备 ID 或形如 ``xpu:x`` 的设备名称。如果 ``device`` 为 None，则 ``device`` 为当前的设备。默认值为 None。


返回
::::::::

None

代码示例
::::::::

COPY-FROM: paddle.device.xpu.reset_max_memory_allocated
