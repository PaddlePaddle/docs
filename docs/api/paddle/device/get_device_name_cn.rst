.. _cn_api_paddle_device_get_device_name:

get_device_name
---------------

.. py:function:: paddle.device.get_device_name(device=None)

获取设备名称。

参数
::::::::::::
    - **device** (int|paddle.CUDAPlace|None) - 设备、设备的 id 或设备的字符串名称，如 ``npu:x``，从中获取设备的属性。 如果设备为 None，则该设备为当前设备，默认值：None。

返回
::::::::::::
    str: 设备名称

代码示例
::::::::::::
COPY-FROM: paddle.device.get_device_name
