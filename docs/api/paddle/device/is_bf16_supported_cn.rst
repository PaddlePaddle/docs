.. _cn_api_paddle_device_is_bf16_supported:

is_bf16_supported
-----------------

.. py:function:: paddle.device.is_bf16_supported(including_emulation: bool = True)

该函数用于检查当前 CUDA 设备是否支持 bfloat16 计算。

参数
:::::::::
- **including_emulation** (bool) - 是否包括模拟支持。默认为 True。True 表示包括模拟支持，False 表示不包括模拟支持。

返回：
::::::::::::
    - bool - 如果设备支持 bfloat16 计算则返回 True，否则返回 False

代码示例
::::::::::::
COPY-FROM: paddle.device.is_bf16_supported
