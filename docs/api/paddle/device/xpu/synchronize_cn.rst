.. _cn_api_ppaddle_device_xpu_synchronize:

synchronize
-------------------------------

.. py:function:: paddle.device.xpu.synchronize(device=None)
等待给定 XPU 设备上的计算完成。

参数
:::::::::

    - **device** (paddle.XPUPlace()|int, 可选) - 设备或设备的 ID。
    - **None** (If device is) - 无
    - **Default** (the device is the current device.) - 无

代码示例
::::::::::::

COPY-FROM: paddle.device.xpu.synchronize
