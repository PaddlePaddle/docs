.. _cn_api_paddle_device_cuda_memory_allocated:

memory_allocated
-------------------------------

.. py:function:: paddle.device.cuda.memory_allocated(device=None)

返回给定设备上当前分配给 Tensor 的显存大小。

.. note::
    Paddle 中分配给 Tensor 的显存块大小会进行 256 字节对齐，因此可能大于 Tensor 实际需要的显存大小。例如，一个 shape 为[1]的 float32 类型 Tensor 会占用 256 字节的显存，即使存储一个 float32 类型数据实际只需要 4 字节。

参数
::::::::

**device** (paddle.CUDAPlace|int|str，可选) - 设备、设备 ID 或形如 ``gpu:x`` 的设备名称。如果 ``device`` 为 None，则 ``device`` 为当前的设备。默认值为 None。


返回
::::::::

一个整数，表示给定设备上当前分配给 Tensor 的显存大小，以字节为单位。

代码示例
::::::::

COPY-FROM: paddle.device.cuda.memory_allocated
