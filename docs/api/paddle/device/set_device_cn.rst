.. _cn_api_set_device:

set_device
-------------------------------

.. py:function:: paddle.device.set_device(device)


Paddle支持包括CPU和GPU在内的多种设备运行，设备可以通过字符串标识符表示，此功能可以指定OP运行的全局设备。

参数
::::::::::::

    - **device** (str)- 此参数确定特定的运行设备，它可以是 ``cpu``、 ``gpu``、 ``xpu``、 ``mlu``、 ``npu``、 ``gpu:x``、 ``xpu:x``、 ``mlu:x`` 或者是 ``npu:x``。其中，``x`` 是GPU、 XPU、 MLU 或者是 NPU 的编号。当 ``device`` 是 ``cpu`` 的时候，程序在CPU上运行，当device是 ``gpu:x`` 的时候，程序在GPU上运行，当device是 ``mlu:x`` 的时候，程序在MLU上运行，当device是 ``npu:x`` 的时候，程序在NPU上运行。

返回
::::::::::::
Place，设置的Place。

代码示例
::::::::::::

COPY-FROM: paddle.device.set_device
