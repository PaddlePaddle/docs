.. _cn_api_paddle_cuda_mem_get_info:

mem_get_info
------------

.. py:function:: paddle.cuda.mem_get_info(device=None)

使用 ``cudaMemGetInfo`` 返回指定设备的全局空闲显存和显存总量。

参数
::::::::::::

    - **device** (DeviceLike) - 指定要查询的设备，可以是形如 "gpu:0" 之类的设备描述字符串，也可以是 `paddle.CUDAPlace(0)` 之类的设备实例。如果为 None（默认值）或未指定设备索引，则返回由 ``paddle.device.get_device()`` 给出的当前设备的统计信息。

返回
:::::::::
返回指定设备的统计信息。

代码示例
::::::::::::
COPY-FROM: paddle.cuda.mem_get_info
