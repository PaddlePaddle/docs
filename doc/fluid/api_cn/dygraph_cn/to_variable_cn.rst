.. _cn_api_fluid_dygraph_to_variable:

to_variable
-------------------------------

**注意：该API仅支持【动态图】模式**

.. py:function:: paddle.fluid.dygraph.to_variable(value, name=None, zero_copy=None)

该函数实现从numpy\.ndarray对象或者Variable对象创建一个 ``Variable`` 类型的对象。

参数：
    - **value** (ndarray|Variable) – 需要转换的numpy\.ndarray或Variable对象，维度可以为多维，数据类型为numpy\.{float16, float32, float64, int16, int32, int64, uint8, uint16}中的一种。
    - **name**  (str, 可选) – 具体用法请参见 :ref:`api_guide_Name` ，一般无需设置，默认值为None。
    - **zero_copy**  (bool, 可选) – 是否与输入的numpy数组共享内存。此参数仅适用于CPUPlace，当它为None时将设置为True。默认值为None。


返回：如果 ``value`` 是numpy\.ndarray对象，返回由numpy\.ndarray对象创建的 ``Tensor`` ，其数据类型和维度与 ``value`` 一致；如果 ``value`` 是Variable对象，返回 ``value`` 。

返回类型：Variable

**代码示例**:

.. code-block:: python
    
    import numpy as np
    import paddle.fluid as fluid

    with fluid.dygraph.guard(fluid.CPUPlace()):
        x = np.ones([2, 2], np.float32)
        y = fluid.dygraph.to_variable(x, zero_copy=False)
        x[0][0] = -1
        y[0][0].numpy()  # array([1.], dtype=float32)
        y = fluid.dygraph.to_variable(x)
        x[0][0] = 0
        y[0][0].numpy()  # array([0.], dtype=float32)

