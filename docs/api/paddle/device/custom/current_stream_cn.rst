.. _cn_api_device_custom_current_stream:

current_stream
-------------------------------

.. py:function:: paddle.device.custom.current_stream(device, device_id=None)

通过 device 返回当前的 custom device stream。


参数
::::::::::::

    - **device** (paddle.CustomPlace()|str) - 希望获取 stream 的设备或者设备类型。
    - **deviec_id** (int，可选) - 希望获取 stream 的设备 ID。如果为 None，则为当前的设备。默认值为 None。

返回
::::::::::::
 CustomDeviceStream，设备的 stream。

代码示例
::::::::::::
COPY-FROM: paddle.device.custom.current_stream
