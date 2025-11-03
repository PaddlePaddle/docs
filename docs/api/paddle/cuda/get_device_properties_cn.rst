.. _cn_api_paddle_cuda_get_device_properties:

get_device_properties
-------------------------------

.. py:function:: paddle.cuda.get_device_properties(device=None)

获取 CUDA 设备的属性信息。

参数
:::::::::
- **device** (int | str | paddle.CUDAPlace | paddle.CustomPlace | None, 可选) - 要查询的目标设备:

  - None: 使用当前设备
  - int: 设备索引 (例如: 0 -> 'gpu:0')
  - str: 设备字符串 (例如: "cuda:0", "gpu:1")
  - CUDAPlace 或 CustomPlace: Paddle 设备对象

返回
:::::::::
DeviceProperties: 包含设备属性的对象，如名称、总内存、计算能力和多处理器数量等。

代码示例
::::::::::::
COPY-FROM: paddle.cuda.get_device_properties
