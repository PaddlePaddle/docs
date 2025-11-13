.. _cn_api_paddle_device_device:

device
------

.. py:function:: paddle.device.device(device=None)

获取或设置当前 CUDA 设备。

参数
::::::::::::
    - **device** (int|str|paddle.Place|None) - 设备、设备的 id 或设备的字符串名称，如 ``npu:x``，从中获取设备的属性。 如果设备为 None，则该设备为当前设备，默认值：None。


代码示例
::::::::::::
COPY-FROM: paddle.device.device
