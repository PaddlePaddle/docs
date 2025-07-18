.. _cn_api_paddle_device_xpu_synchronize:

synchronize
-------------------------------

.. py:function:: paddle.device.xpu.synchronize(device=None)

等待给定的 XPU 设备上的计算完成。


参数
::::::::::::

    - **device** (paddle.XPUPlace()|int，可选) - 设备或者设备 ID。如果为 None，则为当前的设备。默认值为 None。

返回
::::::::::::
None

代码示例
::::::::::::
COPY-FROM: paddle.device.xpu.synchronize
