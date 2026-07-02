.. _cn_api_paddle_device_set_default_device:

set_default_device
-------------------------------

.. py:function:: paddle.device.set_default_device(device=None)

Paddle 支持在各种类型的设备上运行，包括 CPU、GPU、XPU、NPU 和 IPU。
该函数可以设置 OP 运行的全局设备。

参数
:::::::::

    - **device** (str|Place|paddle.device|int|None，可选) - 此参数确定特定的运行设备。可以是 ``cpu``、``gpu``、``xpu``、``npu``、``gpu:x``、``xpu:x``、``npu:x`` 和 ``ipu``，
      其中 ``x`` 是 GPU、XPU 或 NPU 的索引。也可以是 ``paddle.device`` 对象或 int 类型的设备索引。如果为 ``None``，则重置为 CPU。
      默认值为 None。

返回
:::::::::

    无。

代码示例
:::::::::

.. code-block:: pycon

    >>> import paddle
    >>> paddle.device.set_default_device("cpu")
