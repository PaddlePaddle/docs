.. _cn_api_paddle_device_xpu_set_debug_level:

set_debug_level
-------------------------------

.. py:function:: paddle.device.xpu.set_debug_level(level=0)

设置 XPU API DEBUG 功能的等级，打印算子调用信息。默认等级为 0，表示不打印任何调试信息。

参数
::::::::

    - **level** (int) -  Debug 等级。0x1 ( trace )，打印算子调用信息。0x10 ( checksum )，打印张量校验和。0x100 ( dump )，将张量保存为 npy 格式的文件。0x1000 ( profiling )，记录每个算子的执行时间。


返回
::::::::

None

代码示例
::::::::

COPY-FROM: paddle.device.xpu.set_debug_level
