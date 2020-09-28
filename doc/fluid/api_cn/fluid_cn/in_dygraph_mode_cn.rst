.. _cn_api_fluid_in_dygraph_mode:

in_dygraph_mode
-------------------------------

.. py:function:: paddle.in_dynamic_mode()




该接口检查程序是否在动态图模式中运行。
从paddle2.0开始，默认开启动态图模式。
可以通过 ``fluid.dygraph.guard`` 接口开启动态图模式。

注意：
    ``paddle.in_dynamic_mode`` 是 ``fluid.in_dygraph_mode``　的別名, 
    我们推荐使用　``paddle.in_dynamic_mode。

返回：如果程序是在动态图模式下运行的，则返回 ``True``。

返回类型：bool

**示例代码**

.. code-block:: python

    import paddle

    print(paddle.in_dynamic_mode())  # True
    paddle.enable_static()
    print(paddle.in_dynamic_mode())  # False


