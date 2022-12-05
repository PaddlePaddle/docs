.. _cn_api_device_custom_synchronize_device:

synchronize_device
-------------------------------

.. py:function:: paddle.device.custom.synchronize_device(device, device_id=None)

等待给定的设备上的计算完成。


参数
::::::::::::

    - **device** (paddle.CustomPlace()|str) - 希望等待的设备或者设备类型。
    - **deviec_id** (int，可选) - 希望等待的设备 ID。默认值为 0。如果为 None，则为当前的设备。默认值为 None。

返回
::::::::::::
None

代码示例
::::::::::::
COPY-FROM: paddle.device.custom.synchronize_device
