.. _cn_api_device_custom_set_current_device:

set_current_device
-------------------------------

.. py:function:: set_current_device(device, device_id=0)

设置当前设备


参数
::::::::::::

    - **device** (paddle.CustomPlace()|str) - 希望设置的设备或者设备类型。
    - **deviec_id** (int，可选) - 希望设置的设备 ID。默认值为 0。

代码示例
::::::::::::
COPY-FROM: paddle.device.custom.set_current_device
