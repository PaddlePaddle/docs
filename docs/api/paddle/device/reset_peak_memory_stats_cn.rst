.. _cn_api_paddle_device_reset_peak_memory_stats:

reset_peak_memory_stats
-----------------------

.. py:function:: paddle.device.reset_peak_memory_stats(device=None)

重置设备的峰值内存统计信息。

参数
::::::::::::
    - **device** (int|paddle.CUDAPlace|None) - 设备、设备的 id 或设备的字符串名称，如 npu:x'，从中获取设备的属性。 如果设备为 None，则该设备为当前设备，默认值：None。

代码示例
::::::::::::
COPY-FROM: paddle.device.reset_peak_memory_stats
