get_device_capability_cn.rst
.. _cn_api_device_cuda_get_device_capability:

get_device_capability
-------------------------------

.. py:function:: paddle.device.cuda.get_device_capability(device=None)

返回设备的主要CUDA性能和次要CUDA性能。


参数：
    - **device** (paddle.CUDAPlace()|int, 可选) - 希望获取性能的设备或者设备ID。如果为None，则为当前的设备。默认值为None。

返回： tuple(int,int), 设备的主要CUDA性能和次要CUDA性能。

**代码示例**：

        .. code-block:: python

            # required: gpu
            import paddle

            paddle.device.cuda.get_device_capability()

            paddle.device.cuda.get_device_capability(0)

            paddle.device.cuda.get_device_capability(paddle.CUDAPlace(0))
