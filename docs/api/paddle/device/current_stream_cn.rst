.. _cn_api_device_current_stream:

current_stream
-------------------------------

.. py:function:: paddle.device.current_stream(device=None)

通过 device 返回当前的 stream。


参数
::::::::::::

    - **device** (paddle.CUDAPlace|paddle.CustomPlace|str) - 希望获取 stream 的设备或者设备类型。如果为 None，则为当前期望的 place。默认值为 None。

返回
::::::::::::
 paddle.device.Stream，设备的 stream。

代码示例
::::::::::::
COPY-FROM: paddle.device.current_stream
