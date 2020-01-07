.. _cn_api_fluid_dygraph_guard:

guard
-------------------------------

.. py:function:: paddle.fluid.dygraph.guard(place=None)

通过with语句创建一个dygraph运行的context，执行context代码。

参数：
    - **place** (fluid.CPUPlace|fluid.CUDAPlace, 可选) –  动态图执行的设备，可以选择cpu，gpu，如果用户未制定，则根据用户paddle编译的方式来选择运行的设备，如果编译的cpu版本，则在cpu上运行，如果是编译的gpu版本，则在gpu上运行。默认值：None。

返回： None

**代码示例**

.. code-block:: python

    import numpy as np
    import paddle.fluid as fluid

    with fluid.dygraph.guard():
        inp = np.ones([3, 1024], dtype='float32')
        t = fluid.dygraph.base.to_variable(inp)
        fc1 = fluid.Linear(1024, 4, bias_attr=False)
        fc2 = fluid.Linear(4, 4)
        ret = fc1(t)
        dy_ret = fc2(ret)


