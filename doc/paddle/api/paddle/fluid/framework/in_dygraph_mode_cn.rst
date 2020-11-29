.. _cn_api_paddle_in_dynamic_mode:

in_dynamic_mode
-------------------------------

.. py:function:: paddle.in_dynamic_mode()

.. note::
    从2.0.0版本开始，Paddle默认开启动态图模式。

该接口查看paddle当前是否在动态图模式中运行。

可以通过 :ref:`cn_api_paddle_enable_static` 开启静态图模式， :ref:`cn_api_paddle_disable_static` 关闭静态图模式。

返回：如果paddle当前是在动态图模式运行，则返回 ``True`` ，否则返回 ``False``

返回类型：bool

**代码示例**

.. code-block:: python

    import paddle
    print(paddle.in_dynamic_mode())  # True, dynamic mode is turn ON by default since paddle 2.0.0

    paddle.enable_static()
    print(paddle.in_dynamic_mode())  # False, Now we are in static mode

    paddle.disable_static()
    print(paddle.in_dynamic_mode())  # True, Now we are in dynamic mode


