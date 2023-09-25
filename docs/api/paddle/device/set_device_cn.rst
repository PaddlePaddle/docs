.. _cn_api_paddle_device_set_device:

set_device
-------------------------------

.. py:function:: paddle.device.set_device(device)


Paddle 支持包括 CPU 和 GPU 在内的多种设备运行，设备可以通过字符串标识符表示，此功能可以指定 OP 运行的全局设备。

参数
::::::::::::

    - **device** (str)- 此参数确定特定的运行设备，它可以是 ``cpu``、 ``gpu``、 ``xpu``、 ``mlu``、 ``npu``、 ``gpu:x``、 ``xpu:x``、 ``mlu:x`` 或者是 ``npu:x``。其中，``x`` 是 GPU、 XPU、 MLU 或者是 NPU 的编号。当 ``device`` 是 ``cpu`` 的时候，程序在 CPU 上运行，当 device 是 ``gpu:x`` 的时候，程序在 GPU 上运行，当 device 是 ``mlu:x`` 的时候，程序在 MLU 上运行，当 device 是 ``npu:x`` 的时候，程序在 NPU 上运行。

返回
::::::::::::
Place，设置的 Place。

代码示例
::::::::::::

COPY-FROM: paddle.device.set_device
