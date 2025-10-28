.. _cn_api_paddle_device_get_device_properties:

get_device_properties
---------------------

.. py:function:: paddle.device.get_device_properties(device=None)

返回指定设备的属性。

参数
::::::::::::

    - **device** (paddle.CUDAPlace|paddle.CustomPlace|paddle.XPUPlace|str|int，可选) - 设备、设备 ID 或形如 ``gpu:x``、``xpu:x`` 或自定义设备名称的设备字符串。如果为 None，则返回当前设备的属性。默认值为 None。

返回
::::::::::::

    String，指定设备的属性，包括设备名称、主要计算能力、次要计算能力、全局可用内存和设备上的多处理器数量。

代码示例
::::::::::::
COPY-FROM: paddle.device.get_device_properties
