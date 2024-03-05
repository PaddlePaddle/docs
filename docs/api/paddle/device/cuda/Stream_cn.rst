.. _cn_api_paddle_device_cuda_Stream:

Stream
-------------------------------

.. py:class:: paddle.device.cuda.Stream(device=None, priority=None)

CUDA stream 的句柄。

参数
::::::::::::

    - **device** (paddle.CUDAPlace()|int|None，可选) - 希望分配 stream 的设备。如果是 None 或者负数，则设备为当前的设备。如果是正数，则必须小于设备的个数。默认值为 None。
    - **priority** (int|None，可选) - stream 的优先级。优先级可以为 1（高优先级）或者 2（正常优先级）。如果优先级为 None，优先级为 2（正常优先级）。默认值为 None。


代码示例
::::::::::::

COPY-FROM: paddle.device.cuda.Stream



.. warning::
    该 API 未来计划废弃，不推荐使用。
