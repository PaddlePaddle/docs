.. _cn_api_paddle_device_cuda_current_stream:

current_stream
-------------------------------

.. py:function:: paddle.device.cuda.current_stream(device=None)

通过 device 返回当前的 CUDA stream。


参数
::::::::::::

    - **device** (paddle.CUDAPlace()|int，可选) - 希望获取 stream 的设备或者设备 ID。如果为 None，则为当前的设备。默认值为 None。

返回
::::::::::::
 CUDAStream，设备的 stream。

代码示例
::::::::::::
COPY-FROM: paddle.device.cuda.current_stream
