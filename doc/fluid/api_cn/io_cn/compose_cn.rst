.. _cn_api_fluid_io_compose:

composs
-------------------------------

.. py:function:: paddle.fluid.io.compose(*readers, **kwargs)

创建一个数据读取器，输出为输入数据读取器组合到一起的结果，如果输入如下：

（1，2） 3 （4，5）

输出将会为（1，2，3，4，5）。

参数:
    - **readers** – 要组合的输入reader
    - **check_alignment** (bool) - 若为True，将会检查输入readers是否正确的对准，若为False，将不会检查是否对准并且不会跟踪输出，默认为True。

返回：新的数据reader。

Raises：ComposeNotAligned - 输出readers没有对齐，当check_alignment设置为False时将不会raise。