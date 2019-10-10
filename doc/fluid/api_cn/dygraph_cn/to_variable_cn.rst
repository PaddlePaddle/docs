.. _cn_api_fluid_dygraph_to_variable:

to_variable
-------------------------------

.. py:function:: paddle.fluid.dygraph_to_variable(value, block=None, name=None)

该函数实现从numpy\.ndarray对象或者Variable对象创建一个 ``Variable`` 类型的对象。

参数：
    - **value** (ndarray) – 需要转换的numpy\.ndarray对象，维度可以为多维，数据类型为numpy\.{float16, float32, float64, int16, int32, int64, uint8, uint16}中的一种。
    - **block** (fluid.Block, 可选) – Variable所在的Block，默认值为None。
    - **name**  (str, 可选) – 具体用法请参见 :ref:`api_guide_Name` ，一般无需设置，默认值为None。


返回：从指定numpy\.ndarray对象创建的 ``Tensor`` ，数据类型和 ``value`` 一致，返回值维度和 ``value`` 一致

返回类型：Variable

**代码示例**:

.. code-block:: python
    
    import numpy as np
    import paddle.fluid as fluid

    with fluid.dygraph.guard():
        x = np.ones([2, 2], np.float32)
        y = fluid.dygraph.to_variable(x)

