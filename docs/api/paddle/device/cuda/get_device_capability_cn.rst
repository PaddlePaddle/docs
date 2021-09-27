.. _cn_api_device_cuda_get_device_capability:

get_device_capability
-------------------------------

.. py:function:: paddle.device.cuda.get_device_capability(device=None)

返回从CUDA函数`cudaDeviceProp <https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__DEVICE.html#group__CUDART__DEVICE_1g1bf9d625a931d657e08db2b4391170f0>`获取到的定义设备计算能力的主要和次要修订号。


参数：
    - **device** (paddle.CUDAPlace|int, 可选) - 希望获取计算能力的设备或者设备ID。如果为None，则为当前的设备。默认值为None。

返回： tuple(int,int), 设备计算能力的主要和次要修订号。

**代码示例**：

        .. code-block:: python

            # required: gpu
            
            import paddle

            paddle.device.cuda.get_device_capability()

            paddle.device.cuda.get_device_capability(0)

            paddle.device.cuda.get_device_capability(paddle.CUDAPlace(0))
