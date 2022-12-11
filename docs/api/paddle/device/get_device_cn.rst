.. _cn_api_get_device:

get_device
-------------------------------

.. py:function:: paddle.device.get_device()


该功能返回当前程序运行的全局设备，返回的是一个类似于 ``cpu``、 ``gpu:x``、 ``xpu:x``、 ``mlu:x`` 或者 ``npu:x`` 字符串，如果没有设置全局设备，当 cuda 可用的时候返回 ``gpu:0``，当 cuda 不可用的时候返回 ``cpu`` 。

返回
::::::::::::
返回当前程序运行的全局设备。

代码示例
::::::::::::

COPY-FROM: paddle.device.get_device
