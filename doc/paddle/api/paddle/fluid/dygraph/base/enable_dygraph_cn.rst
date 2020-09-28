.. _cn_api_paddle_disable_static:

disable_static
-------------------------------

.. py:function:: paddle.disable_static(place=None)

.. note::
    从2.0.0版本开始，Paddle默认开启动态图模式。

该接口关闭静态图模式。可通过 :ref:`cn_api_paddle_enable_static` 开启静态图模式。


参数：
  - **place** (paddle.CPUPlace|paddle.CUDAPlace，可选) - 动态图运行时的设备。默认值为 ``None`` , 此时，会根据paddle的版本自动判断。

返回：无

**代码示例**

.. code-block:: python

    import paddle
    print(paddle.in_dynamic_mode())  # True, dynamic mode is turn ON by default since paddle 2.0.0

    paddle.enable_static()
    print(paddle.in_dynamic_mode())  # False, Now we are in static mode

    paddle.disable_static()
    print(paddle.in_dynamic_mode())  # True, Now we are in dynamic modes

