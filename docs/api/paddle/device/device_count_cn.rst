.. _cn_api_paddle_device_device_count:

device_count
------------

.. py:function:: paddle.device.device_count(device=None)

返回指定设备类型的可用设备数量。

参数
::::::::::::

    - **device** (paddle.CUDAPlace|paddle.CustomPlace|paddle.XPUPlace|str|int，可选) - 设备、设备 ID 或形如 ``gpu:x``、``xpu:x`` 或自定义设备名称的设备字符串。如果为 None，则返回当前设备类型的可用设备数量。默认值为 None。

返回
::::::::::::

    int，指定设备类型的可用设备数量。

代码示例
::::::::::::
COPY-FROM: paddle.device.device_count
