.. _cn_api_paddle_device_device_guard:

device_guard
-------------------------------

.. py:function:: paddle.device.device_guard(device)

可以切换当前的 device 为输入指定的 device。

.. note::
    该 API 目前仅支持动态图模式。

参数
::::::::::::

    - **device** (PlaceLike) - 指定的 device，可以是形如 "cpu", "gpu:0" 之类的设备描述字符串，也可以是 `paddle.CUDAPlace(0)`, `paddle.CPUPlace()` 之类的设备实例。

代码示例
::::::::::::
COPY-FROM: paddle.device.device_guard
