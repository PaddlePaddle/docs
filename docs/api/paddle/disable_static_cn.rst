.. _cn_api_paddle_disable_static:

disable_static
-------------------------------

.. py:function:: paddle.disable_static(place=None)

.. note::
    从 2.0.0 版本开始，Paddle 默认开启动态图模式。

该接口关闭静态图模式。可通过 :ref:`cn_api_paddle_enable_static` 开启静态图模式。


参数
::::::::::::

  - **place** (paddle.CPUPlace|paddle.CUDAPlace，可选) - 动态图运行时的设备。默认值为 ``None``，此时，会根据 paddle 的版本自动判断。

返回
::::::::::::
无

代码示例
::::::::::::

COPY-FROM: paddle.disable_static
