.. _cn_api_device_cuda_max_memory_allocated_cn:


max_memory_allocated
-------------------------------

.. py:function:: paddle.device.cuda.max_memory_allocated(device=None)

返回给定设备上分配给Tensor的显存峰值。

.. note::
Paddle中分配给Tensor的显存块大小会进行256字节对齐，因此可能大于Tensor实际需要的显存大小。例如，一个shape为[1]的float32类型Tensor会占用256字节的显存，即使存储一个floatt32类型数据实际只需要4字节。

参数
::::::::

**device** (paddle.CUDAPlace|int|str，可选) - 设备、设备ID或形如 ``gpu:x`` 的设备名称。如果 ``device`` 为None，则 ``device`` 为当前的设备。默认值为None。


返回
::::::::

一个整数，表示给定设备上分配给Tensor的显存峰值，以字节为单位。

代码示例
::::::::

COPY-FROM: paddle.device.cuda.max_memory_allocated


