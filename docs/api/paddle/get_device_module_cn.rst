.. _cn_api_paddle_get_device_module:

get_device_module
-------------------------------

.. py:function:: paddle.get_device_module(device=None)

获取指定设备对应的 Paddle 模块。

参数
:::::::::
- **device** (_CustomPlaceLike, 可选) - 要查询的设备，可以是以下类型之一:

  - paddle.Place 对象 (例如 paddle.CUDAPlace(0))
  - 字符串 (例如 "gpu:0", "xpu", "npu")
  - 整数 (设备索引，例如 0 -> "gpu:0")
  - None (使用当前预期设备)

返回
:::::::::
module: 对应的 Paddle 设备模块 (例如 paddle.cuda, paddle.device.xpu)


代码示例
:::::::::
.. code-block:: python

    >>> import paddle
    >>> paddle.get_device_module("gpu:0")
    <module 'paddle.cuda' ...>
