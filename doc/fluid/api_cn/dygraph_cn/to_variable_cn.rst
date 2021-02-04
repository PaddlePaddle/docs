.. _cn_api_fluid_dygraph_to_variable:

to_variable
-------------------------------


.. py:function:: paddle.fluid.dygraph.to_variable(value, name=None, zero_copy=None)


:api_attr: 命令式编程模式（动态图)



该函数实现从tuple、list、numpy\.ndarray、Variable 对象创建一个 ``Variable`` 类型的对象。


参数：
    - **value** (tuple|list|ndarray|Variable|Tensor) – 初始化的数据。可以是tuple、list、numpy\.ndarray、Variable。
      维度可以为多维，数据类型为numpy\.{float16, float32, float64, int16, int32, int64, uint8, uint16, complex64, complex128}中的一种。
    - **name**  (str, 可选) – 具体用法请参见 :ref:`api_guide_Name` ，一般无需设置，默认值为None。
    - **zero_copy**  (bool, 可选) – 是否与输入的numpy数组共享内存。此参数仅适用于CPUPlace，当它为None时将设置为True。默认值为None。
    - **dtype** (str, 可选) - 返回的 ``Variable`` 所需的数据类型。可以是 'bool'，'float16'，'float32'，'float64'，'int8'，'int16'，'int32'，'int64'，'uint8'。默认值: None。


返回：如果 ``value`` 是tuple/list/numpy\.ndarray对象，返回对应numpy\.ndarray对象创建的 ``Tensor`` ；如果 ``value`` 是Variable对象，直接返回 ``value`` 。

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

        c = np.array([2+1j, 2])
        z = fluid.dygraph.to_variable(c)
        z.numpy() # array([2.+1.j, 2.+0.j])
        z.dtype # 'complex128'

        y = fluid.dygraph.to_variable([[0.1, 1.2], [2.2, 3.1], [4.9, 5.2]])
        y.shape     # [3L, 2L]
        y = fluid.dygraph.to_variable(((0.1, 1.2), (2.2, 3.1), (4.9, 5.2)), dtype='int32')
        y.shape     # [3L, 2L]
        y.dtype     # core.VarDesc.VarType.INT32

