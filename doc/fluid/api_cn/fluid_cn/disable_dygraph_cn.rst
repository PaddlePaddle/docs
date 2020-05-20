.. _cn_api_fluid_disable_dygraph:

disable_dygraph
-------------------------------

.. py:function:: paddle.fluid.disable_dygraph()

该接口关闭动态图模式。

返回：无

**示例代码**

.. code-block:: python

    import paddle
    import paddle.fluid as fluid
    import numpy as np
    fluid.enable_dygraph()
    print(fluid.in_dygraph_mode())
    fluid.disable_dygraph()
    print(fluid.in_dygraph_mode())

