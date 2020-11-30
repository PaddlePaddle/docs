.. _cn_api_fluid_enable_dygraph:

enable_dygraph
-------------------------------

.. py:function:: paddle.fluid.enable_dygraph(place=None)

该接口打开动态图模式。

参数：
  - **place** (fluid.CPUPlace 或 fluid.CUDAPlace，可选) - 执行动态图的设备数目。若为None，则设备根据paddle的编译方式决定。默认值为 ``None``。

返回：无

**示例代码**

.. code-block:: python

    import paddle.fluid as fluid
    import numpy as np
    fluid.enable_dygraph()  # Now we are in dygraph mode
    print(fluid.in_dygraph_mode())  # True
    fluid.disable_dygraph()
    print(fluid.in_dygraph_mode())  # False

