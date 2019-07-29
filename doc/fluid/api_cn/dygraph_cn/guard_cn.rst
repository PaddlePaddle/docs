.. _cn_api_fluid_dygraph_guard:

guard
-------------------------------

.. py:function:: paddle.fluid.dygraph.guard(place=None)

创建一个dygraph上下文，用于运行dygraph。

参数：
    - **place** (fluid.CPUPlace|fluid.CUDAPlace|None) – 执行场所

返回： None

**代码示例**

.. code-block:: python

    import numpy as np
    import paddle.fluid as fluid

    with fluid.dygraph.guard():
        inp = np.ones([3, 32, 32], dtype='float32')
        t = fluid.dygraph.base.to_variable(inp)
        fc1 = fluid.FC('fc1', size=4, bias_attr=False, num_flatten_dims=1)
        fc2 = fluid.FC('fc2', size=4)
        ret = fc1(t)
        dy_ret = fc2(ret)


