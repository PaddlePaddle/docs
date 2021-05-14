.. _cn_api_fluid_layers_sqrt:

sqrt
-------------------------------

.. py:function:: paddle.sqrt(x, name=None)




计算输入的算数平方根。

.. math:: 
        out=\sqrt x=x^{1/2}

.. note::
    请确保输入中的数值是非负数。

参数:

    - **x** (Tensor) - 支持任意维度的Tensor。数据类型为float32，float64或float16。
    - **name** (str，可选) – 具体用法请参见 :ref:`api_guide_Name` ，一般无需设置，默认值为None。

返回：返回类型为Tensor， 数据类型同输入一致。

**代码示例**：

.. code-block:: python

    import paddle

    x = paddle.to_tensor([0., 9., 36.])
    y = paddle.sqrt(x)
    print(y) # y=[0., 3., 6.]










