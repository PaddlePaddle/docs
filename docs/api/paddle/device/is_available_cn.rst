.. _cn_api_paddle_device_is_available:

is_available
-------------------------------

.. py:function:: paddle.device.is_available()

检查当前环境中是否有任何支持的设备可用。

该函数检查 Paddle 是否编译了至少一种加速器支持（如 CUDA、XPU、CustomDevice 等），
以及当前系统是否有至少一个该类型的设备可用。

如果有任何支持的设备可用，则返回 True，否则返回 False。

返回
:::::::::
bool: 如果有至少一个可用设备（GPU/XPU/CustomDevice）则返回 True，否则返回 False。

代码示例
::::::::::::
COPY-FROM: paddle.device.is_available
