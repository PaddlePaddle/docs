.. _cn_api_device_synchronize:

synchronize
-------------------------------

.. py:function:: paddle.device.synchronize(device=None, device_id=None)

等待给定的设备上的计算完成。


参数
::::::::::::

    - **device** (paddle.CUDAPlace|paddle.CustomPlace|str) - 希望等待的设备或者设备类型。如果为 None，则为当前期望的 place。默认值为 None。
    - **deviec_id** (int，可选) - 希望获取 stream 的设备 ID。如果为 None，则为当前的设备。默认值为 None。

代码示例
::::::::::::
COPY-FROM: paddle.device.synchronize
