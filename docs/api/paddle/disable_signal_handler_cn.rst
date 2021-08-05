.. _cn_api_fluid_disable_signal_handler:

disable_signal_handler
-------------------------------

.. py:function:: paddle.disable_signal_handler()

关闭Paddle系统信号处理方法

Paddle默认在C++层面注册了系统信号处理方法，用于优化报错信息。
但是一些特定的Python module可能需要使用某些系统信号，引发冲突。
您可以通过调用本函数来关闭Paddle的系统信号处理方法

返回：无

**示例代码**

COPY-FROM: paddle.disable_signal_handler

