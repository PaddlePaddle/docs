.. _cn_api_paddle_device_get_device:

get_device
-------------------------------

.. py:function:: paddle.device.get_device(input=None)


该功能返回当前程序运行的全局设备。未设置全局设备时，CUDA 可用则返回 ``gpu:0``，否则返回 ``cpu``。若传入 Tensor，则返回该 Tensor 所在设备的设备 ID。

参数
::::::::::::

    - **input** (paddle.Tensor|None，可选) - 待查询设备的 Tensor。默认值为 None。

返回
::::::::::::
当 ``input`` 为 Tensor 时，返回 int：CPU Tensor 返回 -1，GPU Tensor 返回其设备 ID。当 ``input`` 不是 Tensor 时，返回当前程序运行设备的名称字符串。

代码示例
::::::::::::

COPY-FROM: paddle.device.get_device
