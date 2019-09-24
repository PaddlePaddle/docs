.. _cn_api_fluid_dygraph_to_variable:

to_variable
-------------------------------

.. py:function:: paddle.fluid.dygraph_to_variable(value, block=None, name=None)

该函数实现从ndarray对象创建一个 ``Variable`` 类型的对象。

参数：
    - **value** (ndarray) – 需要转换的ndarray对象，维度可以为多维，数据类型为np\.float16、np\.float32、np\.float64、np\.int16、np\.int32、np\.int64、np\.uint8、np\.uint16。
    - **block** (fluid.Block, 可选) – Variable所在的Block，默认为None
    - **name**  (str, 可选) – variable的名称，默认为None


返回：从指定ndarray对象创建的 ``Tensor`` ，数据类型和 ``value`` 一致，返回值维度和 ``value`` 一致

返回类型：Variable

**代码示例**:

.. code-block:: python
    
    import numpy as np
    import paddle.fluid as fluid

    with fluid.dygraph.guard():
        x = np.ones([2, 2], np.float32)
        y = fluid.dygraph.to_variable(x)

