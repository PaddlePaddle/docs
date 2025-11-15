.. _cn_api_paddle_device_get_device_capability:

get_device_capability
---------------------

.. py:function:: paddle.device.get_device_capability(device=None)

获取设备计算能力。

参数
::::::::::::
    - **device** (int|str|paddle.Place|None) - 设备、设备的 id 或设备的字符串名称，如 npu:x'，从中获取设备的属性。如果输入 None，则该设备为当前设备。

返回
::::::::::::
    tuple: (主版本号, 次版本号)

代码示例
::::::::::::
COPY-FROM: paddle.device.get_device_capability
