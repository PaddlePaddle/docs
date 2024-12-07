.. _cn_api_paddle_device_cuda_reset_max_memory_allocated:

reset_max_memory_allocated
-------------------------------

.. py:function:: paddle.device.cuda.reset_max_memory_allocated(device=None)

重置特定设备上张量占用的最大GPU内存的跟踪起点。（此函数会调用reset_peak_memory_stats函数，重置CUDA内存分配器的峰值统计信息。）

参数
::::::::

**device** (paddle.CUDAPlace|int|str，可选) - 设备、设备 ID 或形如 ``gpu:x`` 的设备名称。如果 ``device`` 为 None，则 ``device`` 为当前的设备。默认值为 None。


返回
::::::::

None

代码示例
::::::::

COPY-FROM: paddle.device.cuda.reset_max_memory_allocated
