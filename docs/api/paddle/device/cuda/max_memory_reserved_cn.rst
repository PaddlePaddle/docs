.. _cn_api_device_cuda_max_memory_reserved_cn:


max_memory_reserved
-------------------------------

.. py:function:: paddle.device.cuda.max_memory_reserved(device=None)

返回给定设备上由Allocator管理的显存峰值。

参数
::::::::

**device** (paddle.CUDAPlace|int|str，可选) - 设备、设备ID或形如 ``gpu:x`` 的设备名称。如果 ``device`` 为None，则 ``device`` 为当前的设备。默认值为None。


返回
::::::::

一个整数，表示给定设备上当前由Allocator管理的显存峰值，以字节为单位。

代码示例
::::::::

COPY-FROM: paddle.device.cuda.max_memory_reserved


