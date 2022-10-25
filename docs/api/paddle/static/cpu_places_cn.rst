.. _cn_api_fluid_cpu_places:

cpu_places
-------------------------------

.. py:function:: paddle.static.cpu_places(device_count=None)


创建 ``device_count`` 个 ``paddle.CPUPlace`` 对象，并返回所创建的对象列表。

如果 ``device_count`` 为 ``None``，则设备数目将由环境变量 ``CPU_NUM`` 确定。如果未设置 ``CPU_NUM`` 环境变量，则设备数目会默认设为 1，也就是说，``CPU_NUM=1``。
``CPU_NUM`` 表示在当前任务中使用的设备数目。如果 ``CPU_NUM`` 与物理核心数相同，可以加速程序的运行。

参数
:::::::::
  - **device_count** (int，可选) - 设备数目。默认值为 ``None``。

返回
:::::::::
list[paddle.CPUPlace]， ``CPUPlace`` 的列表。

代码示例
:::::::::

COPY-FROM: paddle.static.cpu_places
