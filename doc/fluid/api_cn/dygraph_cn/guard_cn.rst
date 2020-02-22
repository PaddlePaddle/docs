.. _cn_api_fluid_dygraph_guard:

guard
-------------------------------

**注意：该API仅支持【动态图】模式**

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
        inp = np.ones([3, 32, 32], dtype='float32')
        t = fluid.dygraph.base.to_variable(inp)
        fc1 = fluid.FC('fc1', size=4, bias_attr=False, num_flatten_dims=1)
        fc2 = fluid.FC('fc2', size=4)
        ret = fc1(t)
        dy_ret = fc2(ret)


