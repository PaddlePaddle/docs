.. _cn_api_fluid_is_compiled_with_custom_device:

is_compiled_with_custom_device
-------------------------------

.. py:function:: paddle.device.is_compiled_with_custom_device(device_type)

检查 ``whl`` 包是否可以被用来在指定类型的自定义新硬件上运行模型

返回
::::::::::::
bool，支持指定 device_type 则为 True，否则为 False。

代码示例
::::::::::::

COPY-FROM: paddle.device.is_compiled_with_custom_device
