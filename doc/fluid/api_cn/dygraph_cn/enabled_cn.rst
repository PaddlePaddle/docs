.. _cn_api_fluid_dygraph_enabled:

enabled
-------------------------------

.. py:method:: paddle.fluid.dygraph.enabled()

这个函数用于检查程序是否运行在动态图模式。你可以使用 ref:`api_fluid_dygraph_guard` api进入动态图模式。或者使用 ref:`api_fluid_dygraph_enable` 和 ref:`api_fluid_dygraph_disable` api打开、关闭动态图模式。

.. note::
``fluid.dygraph.enabled``实际上调用了``fluid.in_dygraph_mode``，所以推荐使用``fluid.in_dygraph_mode``。

返回: 程序是否运行在动态图模式。

返回类型: bool

**示例代码 1**
  .. code-block:: python

            import paddle.fluid as fluid

            fluid.enable_dygraph()  # Now we are in dygragh mode
            print(fluid.dygraph.enabled())  # True
            fluid.disable_dygraph()
            print(fluid.dygraph.enabled())  # False
