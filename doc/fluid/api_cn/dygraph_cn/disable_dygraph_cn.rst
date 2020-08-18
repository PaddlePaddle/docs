.. _cn_api_fluid_dygraph_disable_dygraph:

disable_dygraph
-------------------------------

.. py:function:: paddle.fluid.dygraph.disable_dygraph()

该接口关闭动态图模式。

.. note::
    推荐使用 :ref:`cn_api_fluid_dygraph_disable_imperative` 。

返回
::::::::::::
无

代码示例
::::::::::::

.. code-block:: python

    import paddle.fluid as fluid
    import numpy as np
    fluid.dygraph.enable_dygraph()  # Now we are in dygraph mode
    print(fluid.in_dygraph_mode())  # True
    fluid.dygraph.disable_dygraph()
    print(fluid.in_dygraph_mode())  # False

