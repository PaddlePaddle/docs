.. _cn_api_paddle_device_cuda_get_device_name:

get_device_name
-------------------------------

.. py:function:: paddle.device.cuda.get_device_name(device=None)

返回从 CUDA 函数 `cudaDeviceProp <https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__DEVICE.html#group__CUDART__DEVICE_1g1bf9d625a931d657e08db2b4391170f0>`_ 获取到的设备名称。


参数
::::::::::
    - **device** (paddle.CUDAPlace|int，可选) - 希望获取名称的设备或者设备 ID。如果 device 为 None（默认），则为当前的设备。

返回
::::::::::
str：设备的名称。


代码示例
::::::::::

COPY-FROM: paddle.device.cuda.get_device_name
