.. _cn_api_paddle_cuda_memory_allocated:

memory_allocated
---------------

.. py:function:: paddle.cuda.memory_allocated(device=None)

返回给定设备上当前分配给 Tensor 的显存大小。

参数
::::::::::::

    - **device** (DeviceLike) - 指定要查询的设备，可以是 "int" 用来表示设备 id，可以是形如 "gpu:0" 之类的设备描述字符串，也可以是 ``paddle.CUDAPlace(0)`` 之类的设备实例。如果为 None（默认值）或未指定设备索引，则返回由 ``paddle.device.get_device()`` 给出的当前设备的统计信息。

返回
::::::::::::

    int, 当前设备上分配的内存总量（字节）。

代码示例
::::::::::::
COPY-FROM: paddle.cuda.memory_allocated
