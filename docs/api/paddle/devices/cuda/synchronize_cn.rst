.. _cn_api_devices_cuda_synchronize:

synchronize
-------------------------------

.. py:function:: paddle.devices.cuda.synchronize(device=None)

等待给定的CUDA 设备上的计算完成。


参数：
    - **device** (paddle.CUDAPlace()|int, 可选) - 设备或者设备ID。如果为None，则为当前的设备。默认值为None。

返回：None

**代码示例**：
COPY-FROM: paddle.devices.cuda.synchronize

