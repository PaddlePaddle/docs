.. _cn_api_fluid_dygraph_enabled:

enabled
-------------------------------

方法
::::::::::::
paddle.fluid.dygraph.enabled()
'''''''''

这个函数用于检查程序是否运行在动态图模式。你可以使用 :ref:`cn_api_fluid_dygraph_guard` api进入动态图模式。或者使用 :ref:`cn_api_fluid_enable_dygraph` 和 :ref:`cn_api_fluid_disable_dygraph` api打开、关闭动态图模式。

注意：`fluid.dygraph.enabled` 实际上调用了 :ref:`cn_api_fluid_in_dygraph_mode` api，所以推荐使用 :ref:`cn_api_fluid_in_dygraph_mode` api。

**返回**
   程序是否运行在动态图模式。

**返回类型**
       bool

**代码示例**

.. code-block:: python

            import paddle.fluid as fluid

            fluid.enable_dygraph()  # Now we are in dygragh mode
            print(fluid.dygraph.enabled())  # True
            fluid.disable_dygraph()
            print(fluid.dygraph.enabled())  # False
