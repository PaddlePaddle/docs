.. _cn_api_fluid_layers_floor:

floor
-------------------------------

.. py:function:: paddle.floor(x, name=None)




向下取整函数。

.. math::
    out = \left \lfloor x \right \rfloor

参数：
    - **x** - 该OP的输入为多维Tensor。数据类型必须为float32或float64。
    - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name` ，一般无需设置，默认值为None。

返回：输出为Tensor，与 ``x`` 维度相同、数据类型相同。

返回类型：Tensor

**代码示例**：

.. code-block:: python

        import paddle

        data1 = paddle.full(shape=[3, 2], fill_value=2.5, dtype='float32') # [[2.5, 2.5], [2.5, 2.5], [2.5, 2.5]]
        data2 = paddle.full(shape=[2, 3], fill_value=-2.5, dtype='float64') # [[-2.5, -2.5, -2.5], [-2.5, -2.5, -2.5]]
        result1 = paddle.floor(data1) # [[2., 2.], [2., 2.], [2., 2.]]
        result2 = paddle.floor(data2) # [[-3., -3., -3.], [-3., -3., -3.]]
        print(result1)
        print(result2)

