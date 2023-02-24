.. _cn_api_paddle_in_dynamic_mode:

in_dynamic_mode
-------------------------------

:: `paddle.in_dynamic_mode() <https://github.com/PaddlePaddle/Paddle/blob/develop/python/paddle/fluid/framework.py>`_

.. note::
    从 2.0.0 版本开始，Paddle 默认开启动态图模式。

该接口查看 paddle 当前是否在动态图模式中运行。

可以通过 :ref:`cn_api_paddle_enable_static` 开启静态图模式，:ref:`cn_api_paddle_disable_static` 关闭静态图模式。

返回
::::::::::::
bool，如果 paddle 当前是在动态图模式运行，则返回 ``True``，否则返回 ``False``。


代码示例
::::::::::::

       .. code-block:: python
            import paddle
            print(paddle.in_dynamic_mode())  # True, dynamic mode is turn ON by default since paddle 2.0.0
            paddle.enable_static()
            print(paddle.in_dynamic_mode())  # False, Now we are in static graph mode
            paddle.disable_static()
            print(paddle.in_dynamic_mode())  # True, Now we are in dynamic mode
