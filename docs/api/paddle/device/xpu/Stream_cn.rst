.. _cn_api_paddle_device_xpu_Stream:

Stream
-------------------------------

.. py:class:: paddle.device.xpu.Stream(device=None)

XPU stream 的句柄。

参数
::::::::::::

    - **device** (paddle.XPUPlace()|int|None，可选) - 希望分配 stream 的设备。如果是 None 或者负数，则设备为当前的设备。如果是正数，则必须小于设备的个数。默认值为 None。


代码示例
::::::::::::

COPY-FROM: paddle.device.xpu.Stream



.. warning::
    该 API 未来计划废弃，不推荐使用。
