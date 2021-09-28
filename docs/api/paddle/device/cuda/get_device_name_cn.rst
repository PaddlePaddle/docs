.. _cn_api_device_cuda_get_device_name:

get_device_name
-------------------------------

.. py:function:: paddle.device.cuda.get_device_name(device=None)

返回从CUDA函数`cudaDeviceProp <https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__DEVICE.html#group__CUDART__DEVICE_1g1bf9d625a931d657e08db2b4391170f0>`_ 获取到的设备名称。


参数：
    - **device** (paddle.CUDAPlace|int, 可选) - 希望获取名称的设备或者设备ID。如果为None，则为当前的设备。默认值为None。

返回： str, 设备的名称

**代码示例**：

        .. code-block:: python

            # required: gpu
            
            import paddle

            paddle.device.cuda.get_device_name()

            paddle.device.cuda.get_device_name(0)

            paddle.device.cuda.get_device_name(paddle.CUDAPlace(0))
