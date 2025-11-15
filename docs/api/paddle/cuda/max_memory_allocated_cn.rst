.. _cn_api_paddle_cuda_max_memory_allocated:

max_memory_allocated
--------------------

.. py:function:: paddle.cuda.max_memory_allocated(device=None)

返回给定设备上分配给 Tensor 的显存峰值统计。

.. note::
    Paddle 中分配给 Tensor 的显存块大小会进行 256 字节对齐，因此可能大于 Tensor 实际需要的显存大小。例如，一个 shape 为[1]的 float32 类型 Tensor 会占用 256 字节的显存，即使存储一个 float32 类型数据实际只需要 4 字节。
    与 :ref:`cn_api_paddle_device_max_memory_allocated` 功能一致

参数
::::::::::::
    - **device** (int|paddle.Place|str|None) - 设备、设备的 id 或设备的字符串名称，如 ``npu:x``，从中获取设备的属性。 如果设备为 None，则该设备为当前设备，默认值：None。

返回
::::::::::::
    int: 最大已分配内存量（字节）

代码示例
::::::::::::
COPY-FROM: paddle.cuda.max_memory_allocated
