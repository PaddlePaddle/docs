.. _cn_api_device_cuda_current_stream:

current_stream
-------------------------------

.. py:function:: paddle.device.cuda.current_stream(device=None)

通过device 返回当前的CUDA stream。


参数：
    - **device** (paddle.CUDAPlace()|int, 可选) - 希望获取stream的设备或者设备ID。如果为None，则为当前的设备。默认值为None。

返回： CUDAStream, 设备的stream

**代码示例**：
COPY-FROM: paddle.device.cuda.current_stream

