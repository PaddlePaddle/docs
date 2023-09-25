.. _cn_api_paddle_device_cuda_get_device_properties:


get_device_properties
-------------------------------

.. py:function:: paddle.device.cuda.get_device_properties(device)

返回给定的设备属性。

参数
::::::::

**device** (paddle.CUDAPlace or int or str) - 设备、设备 ID 和类似于 ``gpu:x`` 的设备名称。如果 ``device`` 为空，则 ``device`` 为当前的设备。默认值为 None。


返回
::::::::

_gpuDeviceProperties：设备属性，包括标识设备的 ASCII 字符串、设备计算能力的主版本号以及次版本号、全局显存总量、设备上多处理器的数量。


代码示例
::::::::

COPY-FROM: paddle.device.cuda.get_device_properties
