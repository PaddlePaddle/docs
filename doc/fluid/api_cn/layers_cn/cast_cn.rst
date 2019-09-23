.. _cn_api_fluid_layers_cast:

cast
-------------------------------

.. py:function:: paddle.fluid.layers.cast(x,dtype)

该OP将 ``x`` 的数据类型转换为 ``dtype`` 并输出。支持输出和输入的数据类型相同。

参数：
    - **x** (Variable) - 输入的多维 ``Tensor`` ，支持的数据类型为：bool、float16、float32、float64、uint8、int32、int64
    - **dtype** (str|np.dtype|core.VarDesc.VarType) - 输出的数据类型

返回：``Tensor`` ，维度与 ``x`` 相同，数据类型为 ``dtype``

返回类型：Variable

**代码示例**：

.. code-block:: python

    import paddle.fluid as fluid
    data = fluid.layers.data(name='x', shape=[13], dtype='float32')
    result = fluid.layers.cast(x=data, dtype='float64')
