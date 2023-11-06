.. _cn_api_paddle_static_device_guard:

device_guard
-------------------------------

.. py:function:: paddle.static.device_guard(device=None)

.. note::
    该 API 仅支持静态图模式。

一个用于指定 OP 运行设备的上下文管理器。

参数
::::::::::::

    - **device** (str|None) – 指定上下文中使用的设备。它可以是 ``cpu``、 ``gpu``、 ``gpu:x``，其中 ``x`` 是 GPU 的编号。当它被设置为 ``cpu`` 或者 ``gpu`` 时，创建在该上下文中的 OP 将被运行在 CPUPlace 或者 CUDAPlace 上。若设置为 ``gpu``，同时程序运行在单卡模式下，设备的索引将与执行器的设备索引保持一致，默认值：None，在该上下文中的 OP 将被自动地分配设备。

代码示例
::::::::::::

COPY-FROM: paddle.static.device_guard
