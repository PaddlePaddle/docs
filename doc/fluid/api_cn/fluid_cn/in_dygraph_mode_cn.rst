.. _cn_api_fluid_in_dygraph_mode:

in_dygraph_mode
-------------------------------

.. py:function:: paddle.fluid.in_dygraph_mode()

检查程序状态(tracer) - 是否在dygraph模式中运行

返回：如果Program是在动态图模式下运行的则为True。

返回类型：out(boolean)

**示例代码**

.. code-block:: python

    import paddle.fluid as fluid
    if fluid.in_dygraph_mode():
        pass


