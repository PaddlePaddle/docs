.. _cn_api_paddle_enable_static:

enable_static
-------------------------------

.. py:function:: paddle.enable_static()

.. note::
    从2.0.0版本开始，Paddle默认开启动态图模式。

该接口开启静态图模式。可通过 :ref:`cn_api_paddle_disable_static` 关闭静态图模式。


返回：无

**代码示例**

.. code-block:: python

    import paddle
    print(paddle.in_dynamic_mode())  # True, dynamic mode is turn ON by default since paddle 2.0.0

    paddle.enable_static()
    print(paddle.in_dynamic_mode())  # False, Now we are in static mode

    paddle.disable_static()
    print(paddle.in_dynamic_mode())  # True, Now we are in dynamic mode
