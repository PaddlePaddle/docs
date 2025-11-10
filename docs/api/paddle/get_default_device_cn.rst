.. _cn_api_paddle_get_default_device:

get_default_device
-------------------------------

.. py:function:: paddle.get_default_device()

获取程序当前运行的全局设备信息。

返回一个表示当前设备的字符串，格式可能是：
返回一个表示当前设备的字符串，格式可能是：

- 'cpu'
- 'gpu:x' (CUDA 设备)
- 'xpu:x' (XPU 设备)
- 'npu:x' (NPU 设备)

如果全局设备未明确设置，将根据以下规则返回：
如果全局设备未明确设置，将根据以下规则返回：

- 当 CUDA 可用时返回 'gpu:x'
- 当 CUDA 不可用时返回 'cpu'

返回
:::::::::
str: 表示当前设备的字符串

代码示例
:::::::::
.. code-block:: python

    >>> import paddle
    >>> device = paddle.get_default_device()
