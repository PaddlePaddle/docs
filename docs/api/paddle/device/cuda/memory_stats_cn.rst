.. _cn_api_paddle_device_cuda_memory_stats:

memory_stats
-------------------------------

.. py:function:: paddle.device.cuda.memory_stats(device=None)

返回包含给定设备CUDA内存分配器统计信息的字典。

参数
::::::::

**device** (paddle.CUDAPlace|int|str，可选) - 设备、设备 ID 或形如 ``gpu:x`` 的设备名称。如果 ``device`` 为 None，则 ``device`` 为当前的设备。默认值为 None。


返回
::::::::

包含给定设备CUDA内存分配器统计信息的字典。字典的键值对如下：

"memory.allocated.current"：分配给张量的GPU内存的当前大小。
"memory.allocated.peak"：分配给张量的GPU内存的峰值大小。
"memory.reserved.current"：分配器持有的GPU内存的当前大小。
"memory.reserved.peak"：分配器持有的GPU内存的峰值大小。

代码示例
::::::::

COPY-FROM: paddle.device.cuda.memory_stats
