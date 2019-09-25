.. _cn_api_fluid_layers_floor:

floor
-------------------------------

.. py:function:: paddle.fluid.layers.floor(x, name=None)

向下取整函数。

.. math::
    out = \left \lfloor x \right \rfloor

参数：
    - **x** - 该OP的输入为多维Tensor。数据类型必须为float32或float64。
    - **name** (str, 可选) - 该参数供开发人员打印调试信息时使用，具体用法请参见 :ref:`api_guide_Name`，默认值为None。

返回：输出为Tensor，与 ``x`` 维度相同、数据类型相同。

返回类型：Variable

**代码示例**：

.. code-block:: python

        import paddle.fluid as fluid
        data1 = fluid.layers.fill_constant(shape=[3, 2], value=2.5, dtype='float32') # [[2.5, 2.5], [2.5, 2.5], [2.5, 2.5]]
        data2 = fluid.layers.fill_constant(shape=[2, 3], value=-2.5, dtype='float64') # [[-2.5, -2.5, -2.5], [-2.5, -2.5, -2.5]]
        result1 = fluid.layers.floor(data1) # [[2., 2.], [2., 2.], [2., 2.]]
        result2 = fluid.layers.floor(data2) # [[-3., -3., -3.], [-3., -3., -3.]]
