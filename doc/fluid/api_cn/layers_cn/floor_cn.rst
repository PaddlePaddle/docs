.. _cn_api_fluid_layers_floor:

floor
-------------------------------

.. py:function:: paddle.fluid.layers.floor(x, name=None)

:alias_main: paddle.floor
:alias: paddle.floor,paddle.tensor.floor,paddle.tensor.math.floor
:old_api: paddle.fluid.layers.floor



向下取整函数。

.. math::
    out = \left \lfloor x \right \rfloor

参数：
    - **x** - 该OP的输入为多维Tensor。数据类型必须为float32或float64。
    - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name` ，一般无需设置，默认值为None。

返回：输出为Tensor，与 ``x`` 维度相同、数据类型相同。

返回类型：Variable

**代码示例**：

.. code-block:: python

    import paddle
    import paddle.fluid as fluid
    data1 = paddle.full(shape=[3, 2], fill_value=2.5, dtype='float32', device=
        None, stop_gradient=True)
    data2 = paddle.full(shape=[2, 3], fill_value=-2.5, dtype='float64', device=
        None, stop_gradient=True)
    result1 = paddle.floor(data1)
    result2 = paddle.floor(data2)

